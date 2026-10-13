"""These network tests use only an in-process localhost HTTP server and stdlib."""
import hashlib
import importlib.util
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("range_download", Path(__file__).parents[1] / "scripts" / "range_download.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
download = module.download
PAYLOAD = bytes(range(256)) * 37 + b"last partial chunk"


class RangeHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[int, int, str]]] = []
    bad_range = False
    redirect_count = 0
    expire_signed_once = False
    truncate_once = False

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.startswith("/redirect"):
            type(self).redirect_count += 1
            self.send_response(302)
            self.send_header("Location", "/signed?Policy=example&Signature=example")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            self.send_error(400)
            return
        start, end = map(int, match.groups())
        if self.path.startswith("/signed") and self.expire_signed_once and end > start:
            type(self).expire_signed_once = False
            self.send_error(403, "Signed URL expired")
            return
        self.requests.append((start, end, self.path))
        body = PAYLOAD[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start + int(self.bad_range)}-{end}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.truncate_once:
            type(self).truncate_once = False
            self.wfile.write(body[:len(body) // 2])
            self.close_connection = True
            return
        self.wfile.write(body)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.target = Path(self.directory.name) / "model.bin"
        RangeHandler.requests = []
        RangeHandler.bad_range = False
        RangeHandler.redirect_count = 0
        RangeHandler.expire_signed_once = False
        RangeHandler.truncate_once = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/model?source=test"
        self.sha = hashlib.sha256(PAYLOAD).hexdigest()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.directory.cleanup()

    def test_parallel_download_and_existing_file_verification(self):
        download(self.url, self.target, len(PAYLOAD), self.sha, workers=4, chunk_size=1024)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(len(RangeHandler.requests), 10)
        self.assertTrue(all("download_chunk_start=" in path and "source=test" in path
                            for _, _, path in RangeHandler.requests))
        download(self.url, self.target, len(PAYLOAD), self.sha, workers=4, chunk_size=1024)
        self.assertEqual(len(RangeHandler.requests), 10)

    def test_resume_exact_parts_and_replace_incorrect_size(self):
        parts = self.target.with_name(self.target.name + ".parts")
        parts.mkdir()
        manifest = {"url": self.url, "size": len(PAYLOAD), "sha256": self.sha, "chunk_size": 1024}
        (parts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (parts / "00000000").write_bytes(PAYLOAD[:1024])
        (parts / "00000001").write_bytes(b"incomplete")
        download(self.url, self.target, len(PAYLOAD), self.sha, workers=3, chunk_size=1024)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(len(RangeHandler.requests), 9)
        self.assertFalse(any(start == 0 for start, _, _ in RangeHandler.requests))

    def test_sha_mismatch_does_not_replace_target(self):
        self.target.write_bytes(b"preserve old file")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            download(self.url, self.target, len(PAYLOAD), "0" * 64, workers=3, chunk_size=4096)
        self.assertEqual(self.target.read_bytes(), b"preserve old file")

    def test_wrong_content_range_retried_and_never_published(self):
        RangeHandler.bad_range = True
        with patch.object(module, "_wait_retry", return_value=False), self.assertRaisesRegex(RuntimeError, "failed after 8 attempts"):
            download(self.url, self.target, len(PAYLOAD), self.sha, workers=1, chunk_size=len(PAYLOAD))
        self.assertEqual(len(RangeHandler.requests), 8)
        self.assertFalse(self.target.exists())

    def test_signed_redirect_reused_without_query_mutation(self):
        self.url = self.url.replace("/model?", "/redirect?")
        with patch.object(module, "_is_hf_source", return_value=True):
            download(self.url, self.target, len(PAYLOAD), self.sha, workers=4, chunk_size=1024)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(RangeHandler.redirect_count, 1)
        self.assertEqual(len(RangeHandler.requests), 11)  # one-byte probe plus ten parts
        self.assertTrue(all(path == "/signed?Policy=example&Signature=example"
                            for _, _, path in RangeHandler.requests))

    def test_expired_signed_redirect_refreshed(self):
        self.url = self.url.replace("/model?", "/redirect?")
        RangeHandler.expire_signed_once = True
        with patch.object(module, "_is_hf_source", return_value=True), patch.object(module, "_wait_retry", return_value=False):
            download(self.url, self.target, len(PAYLOAD), self.sha, workers=1, chunk_size=1024)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(RangeHandler.redirect_count, 2)

    def test_truncated_ranges_resume_validated_prefix_with_each_transport(self):
        payload = bytes(range(256)) * 1024
        transports = ["urllib"]
        if importlib.util.find_spec("httpx"):
            transports.append("httpx")
        for transport in transports:
            with self.subTest(transport=transport), patch.dict(globals(), {"PAYLOAD": payload}), patch.object(module, "_wait_retry", return_value=False):
                RangeHandler.requests = []
                RangeHandler.truncate_once = True
                target = self.target.with_name(f"{transport}.bin")
                download(self.url, target, len(payload), hashlib.sha256(payload).hexdigest(),
                         workers=1, chunk_size=len(payload), transport=transport)
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual([start for start, _, _ in RangeHandler.requests], [0, len(payload) // 2])


if __name__ == "__main__":
    unittest.main()
