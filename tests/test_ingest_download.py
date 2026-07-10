"""Tests for evaluator video URL ingestion."""

import functools
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.ingest import download_video


class QuietHandler(SimpleHTTPRequestHandler):
    """Serve fixtures without polluting test output."""

    def log_message(self, format: str, *args: object) -> None:
        return


class DownloadTests(unittest.TestCase):
    """Verify remote clips are downloaded once and cached."""

    def test_download_video_fetches_and_reuses_cached_content(self) -> None:
        with (
            tempfile.TemporaryDirectory() as source_dir,
            tempfile.TemporaryDirectory() as cache_dir,
        ):
            source = Path(source_dir) / "clip.mp4"
            source.write_bytes(b"fake-video-content")
            handler = functools.partial(QuietHandler, directory=source_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/clip.mp4"
                first = download_video(url, Path(cache_dir))
                source.write_bytes(b"changed-after-first-download")
                second = download_video(url, Path(cache_dir))
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), b"fake-video-content")


if __name__ == "__main__":
    unittest.main()
