"""Resumable, validated HTTP ranges with optional pooling and a stdlib fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def _chunk_url(url: str, start: int, end: int, attempt: int) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.extend((("download_chunk_start", str(start)), ("download_chunk_end", str(end)),
                  ("download_chunk_attempt", str(attempt))))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def _wait_retry(cancelled: threading.Event, seconds: float) -> bool:
    return cancelled.wait(seconds)


def _is_hf_source(url: str) -> bool:
    return urllib.parse.urlsplit(url).hostname in {"huggingface.co", "hf.co"}


class _RedirectCache:
    """Share one signed CDN redirect without modifying its signature parameters."""

    def __init__(self, origin: str, size: int):
        self.origin = origin
        self.size = size
        self.enabled = _is_hf_source(origin)
        self.cached: str | None = None
        self.lock = threading.Lock()

    def resolve(self) -> str | None:
        if not self.enabled:
            return None
        with self.lock:
            if self.cached is not None:
                return self.cached
            probe_url = _chunk_url(self.origin, 0, 0, 0) + f"&download_redirect_refresh={time.time_ns()}"
            request = urllib.request.Request(probe_url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity",
                                                                  "User-Agent": "lecture-summary-range-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get("Content-Range", "").strip() != f"bytes 0-0/{self.size}":
                    raise ValueError("Redirect probe returned an incorrect range")
                if len(response.read(2)) != 1:
                    raise ValueError("Redirect probe returned an incorrect body size")
                candidate = response.url
            # HF normally redirects to a CDN host. If it does not, retain the
            # original per-range cache-busting path instead of caching the probe.
            if candidate == probe_url:
                self.enabled = False
                return None
            self.cached = candidate
            print(f"Using shared CDN redirect: {urllib.parse.urlsplit(candidate).hostname}", flush=True)
            return candidate

    def invalidate(self, failed_url: str):
        with self.lock:
            # A failure from an old connection must not discard a fresh URL.
            if self.cached == failed_url:
                self.cached = None


class _HTTPTransport:
    """Optional HTTP/1.1 connection pooling; stdlib remains a supported fallback."""

    def __init__(self, workers: int, preference: str):
        if preference not in {"auto", "urllib", "httpx"}:
            raise ValueError("transport must be auto, urllib or httpx")
        self.client = None
        if preference != "urllib":
            try:
                import httpx
            except ImportError:
                if preference == "httpx":
                    raise RuntimeError("httpx transport requested but httpx is not installed") from None
            else:
                # Use the same OS certificate trust as urllib, including Windows roots.
                self.client = httpx.Client(
                    verify=ssl.create_default_context(), follow_redirects=True, trust_env=True, timeout=90.0,
                    limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers,
                                       keepalive_expiry=90.0),
                )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.client is not None:
            self.client.close()

    @contextmanager
    def open(self, url: str, headers: dict):
        if self.client is not None:
            with self.client.stream("GET", url, headers=headers) as response:
                if response.status_code >= 400:
                    raise urllib.error.HTTPError(str(response.url), response.status_code,
                                                 response.reason_phrase, response.headers, None)
                yield response.status_code, response.headers, response.iter_raw(chunk_size=64 * 1024)
        else:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                def chunks():
                    while block := response.read(64 * 1024):
                        yield block
                yield response.status, response.headers, chunks()


def download(url: str, target: Path, size: int, sha256: str | None = None,
             workers: int = 12, chunk_size: int = 8 * 1024 * 1024, transport: str = "auto") -> Path:
    """Fetch disjoint ranges, resume exact-sized parts, validate, then atomically publish.

    Parts remain available after failure and after success. A SHA256 mismatch leaves
    the assembled .partial file for diagnosis and never overwrites the target.
    """
    target = Path(target)
    if size <= 0 or workers <= 0 or chunk_size <= 0:
        raise ValueError("size, workers and chunk_size must be positive")
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) URLs are supported")
    if sha256 is not None:
        sha256 = sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size and (sha256 is None or _digest(target) == sha256):
        print(f"Already complete: {target.name} ({size:,} bytes)", flush=True)
        return target

    parts_dir = target.with_name(target.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = parts_dir / "manifest.json"
    manifest = {"url": url, "size": size, "sha256": sha256, "chunk_size": chunk_size}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing parts use different download parameters; choose another target or parts directory")
    else:
        # Without a manifest, existing parts have no trustworthy source identity.
        if any(parts_dir.iterdir()):
            raise ValueError("Existing parts directory has no manifest; choose another target")
        manifest_tmp = parts_dir / "manifest.partial"
        manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)

    ranges = [(index, start, min(start + chunk_size, size) - 1)
              for index, start in enumerate(range(0, size, chunk_size))]
    completed_bytes = 0
    pending = []
    for index, start, end in ranges:
        path = parts_dir / f"{index:08d}"
        if path.is_file() and path.stat().st_size == end - start + 1:
            completed_bytes += end - start + 1
        else:
            pending.append((index, start, end))
    started = time.monotonic()
    print(f"{target.name}: {completed_bytes:,}/{size:,} bytes cached; {len(pending)} ranges to fetch", flush=True)
    cancelled = threading.Event()
    redirect = _RedirectCache(url, size)

    def fetch(item):
        index, start, end = item
        expected = end - start + 1
        part = parts_dir / f"{index:08d}"
        partial = part.with_suffix(".partial")
        bypass_redirect = False
        for attempt in range(8):
            if cancelled.is_set():
                raise RuntimeError("Download cancelled after another range failed")
            cached_url = None
            try:
                existing = partial.stat().st_size if partial.is_file() else 0
                if existing > expected:
                    partial.unlink()
                    existing = 0
                if existing == expected:
                    os.replace(partial, part)
                    return expected
                request_start = start + existing
                cached_url = None if bypass_redirect else redirect.resolve()
                request_url = cached_url or _chunk_url(url, request_start, end, attempt)
                headers = {"Range": f"bytes={request_start}-{end}", "Accept-Encoding": "identity",
                           "User-Agent": "lecture-summary-range-downloader/1.0"}
                with http.open(request_url, headers) as (status, response_headers, chunks):
                    if status != 206:
                        raise ValueError(f"Expected HTTP 206, received {status}")
                    content_range = response_headers.get("Content-Range", "").strip()
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or tuple(map(int, match.groups())) != (request_start, end, size):
                        # Some intermediary caches ignore Range; retry this part
                        # through the original URL with its unique cache key.
                        if cached_url:
                            bypass_redirect = True
                        raise ValueError(f"Incorrect Content-Range: {content_range!r}")
                    content_length = response_headers.get("Content-Length")
                    if content_length is not None and int(content_length) != expected - existing:
                        raise ValueError("Incorrect Content-Length")
                    if response_headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Unexpected Content-Encoding")
                    count = existing
                    with partial.open("ab") as output:
                        for block in chunks:
                            count += len(block)
                            if count > expected:
                                raise ValueError("Range body exceeded requested size")
                            output.write(block)
                        if count != expected:
                            raise ValueError(f"Truncated range: {count}/{expected} bytes")
                        output.flush()
                        os.fsync(output.fileno())
                os.replace(partial, part)
                return expected
            except Exception as exc:
                # Bytes are written only after exact Content-Range validation.
                # Preserve that prefix when the connection closes mid-body.
                retained = partial.stat().st_size if partial.is_file() else 0
                if retained > expected:
                    partial.unlink()
                if isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 403}:
                    failed_url = cached_url or getattr(exc, "url", None)
                    if failed_url:
                        redirect.invalidate(failed_url)
                if attempt == 7:
                    raise RuntimeError(f"Range {start}-{end} failed after 8 attempts ({type(exc).__name__}: {exc})") from exc
                print(f"Retry range {index + 1}/{len(ranges)}: {type(exc).__name__}: {exc} "
                      f"({attempt + 1}/8, {retained:,} bytes retained)", flush=True)
                # Spread retries from concurrent connections instead of reconnecting in lockstep.
                delay = min(2 ** attempt, 20) + random.uniform(0, min(2 ** attempt, 10))
                if _wait_retry(cancelled, delay):
                    raise RuntimeError("Download cancelled") from exc
        raise AssertionError("Unreachable retry state")

    with _HTTPTransport(workers, transport) as http, ThreadPoolExecutor(max_workers=workers, thread_name_prefix="range-fetch") as executor:
        futures = [executor.submit(fetch, item) for item in pending]
        try:
            for future in as_completed(futures):
                completed_bytes += future.result()
                elapsed = max(0.001, time.monotonic() - started)
                print(f"{target.name}: {completed_bytes:,}/{size:,} bytes ({100 * completed_bytes / size:.1f}%), "
                      f"elapsed {elapsed:.0f}s", flush=True)
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise

    assembled = target.with_name(target.name + ".partial")
    checksum = hashlib.sha256()
    total = 0
    with assembled.open("wb") as output:
        for index, start, end in ranges:
            part = parts_dir / f"{index:08d}"
            if part.stat().st_size != end - start + 1:
                raise ValueError(f"Part {index} changed during assembly")
            with part.open("rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    output.write(block)
                    checksum.update(block)
                    total += len(block)
        output.flush()
        os.fsync(output.fileno())
    if total != size:
        raise ValueError(f"Assembled size mismatch: {total}/{size}")
    actual_sha = checksum.hexdigest()
    if sha256 is not None and actual_sha != sha256:
        raise ValueError(f"SHA256 mismatch: expected {sha256}, received {actual_sha}; target was not replaced")
    os.replace(assembled, target)
    print(f"Complete: {target.name}; SHA256 {actual_sha}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--transport", choices=("auto", "urllib", "httpx"), default="auto")
    args = parser.parse_args()
    download(args.url, args.target, args.size, args.sha256, args.workers, args.chunk_size, args.transport)


if __name__ == "__main__":
    main()
