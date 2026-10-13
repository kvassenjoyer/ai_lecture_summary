"""Download exact official HF checkpoints with resumable ranges and immutable hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from filelock import FileLock
from range_download import download

MODELS = ("openai/whisper-large-v3-turbo", "Qwen/Qwen3.5-4B")
SUFFIXES = (".json", ".safetensors", ".txt", ".model", ".jinja", ".tiktoken")


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def small_file(repo: str, revision: str, entry: dict, folder: Path) -> None:
    name = entry["rfilename"]
    target = folder / name
    if target.is_file() and target.stat().st_size == entry["size"]:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                body = response.read()
            if len(body) != entry["size"]:
                raise ValueError(f"Wrong download size: {name}")
            blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            if blob != entry["blobId"]:
                raise ValueError(f"Wrong Git blob checksum: {name}")
            partial = target.with_name(target.name + ".partial")
            partial.write_bytes(body)
            partial.replace(target)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("models"))
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--model", action="append", help="Download only this repository; repeat for multiple models")
    args = parser.parse_args()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"models": []}
    manifest["downloaded_at"] = datetime.now(UTC).isoformat()
    for repo in args.model or MODELS:
        info = fetch_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
        revision = info["sha"]
        print(f"Downloading {repo} at {revision}", flush=True)
        cache = args.cache_root / "hub" / ("models--" + repo.replace("/", "--"))
        folder = cache / "snapshots" / revision
        folder.mkdir(parents=True, exist_ok=True)
        selected = [entry for entry in info["siblings"] if entry["rfilename"].endswith(SUFFIXES)]
        small = [entry for entry in selected if "lfs" not in entry]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(partial(small_file, repo, revision, folder=folder), small))
        for entry in sorted(selected, key=lambda item: item["rfilename"].endswith(".safetensors")):
            if "lfs" in entry:
                download(f"https://huggingface.co/{repo}/resolve/{revision}/{entry['rfilename']}",
                         folder / entry["rfilename"], entry["size"], entry["lfs"]["sha256"],
                         workers=args.workers)
        refs = cache / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(revision, encoding="utf-8")
        files = []
        for entry in selected:
            file = folder / entry["rfilename"]
            record = {"name": entry["rfilename"], "bytes": file.stat().st_size}
            if args.verify_hashes:
                record["sha256"] = sha256(file)
            files.append(record)
        with FileLock(str(manifest_path) + ".lock"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"models": []}
            manifest["downloaded_at"] = datetime.now(UTC).isoformat()
            manifest["models"] = [item for item in manifest["models"] if item["repo_id"] != repo]
            manifest["models"].append({"repo_id": repo, "revision": revision,
                                       "snapshot": str(folder.resolve()), "files": files})
            temporary_manifest = manifest_path.with_suffix(".partial")
            temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary_manifest.replace(manifest_path)
        print(f"Ready: {repo} ({sum(f['bytes'] for f in files) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
