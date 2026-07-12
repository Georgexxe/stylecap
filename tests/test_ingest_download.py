"""Tests for evaluator video URL ingestion."""

import functools
import socket
import subprocess
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from src import ingest
from src.ingest import DownloadError, _evenly_cap, download_video, validate_video_url


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
                first = download_video(url, Path(cache_dir), allow_private=True)
                source.write_bytes(b"changed-after-first-download")
                second = download_video(url, Path(cache_dir), allow_private=True)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), b"fake-video-content")

    def test_evenly_cap_preserves_first_and_last_evidence(self) -> None:
        frames = [f"frame-{index}" for index in range(25)]
        selected = _evenly_cap(frames, 8)

        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0], "frame-0")
        self.assertEqual(selected[-1], "frame-24")

    def test_sampling_fps_targets_the_frame_budget(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="12.0\n",
            stderr="",
        )
        with patch.object(ingest.subprocess, "run", return_value=completed):
            fps = ingest._sampling_fps("clip.mp4")

        self.assertEqual(fps, ingest.config.MAX_FRAMES / 12.0)

    def test_validate_video_url_rejects_private_targets(self) -> None:
        with self.assertRaises(DownloadError):
            validate_video_url("http://127.0.0.1/private.mp4")

    def test_validate_video_url_rejects_non_http_schemes(self) -> None:
        with self.assertRaises(DownloadError):
            validate_video_url("file:///tmp/private.mp4")

    @patch("src.ingest.socket.getaddrinfo")
    def test_validate_video_url_accepts_public_targets(self, getaddrinfo: Mock) -> None:
        getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
        ]
        validate_video_url("https://media.example.com/clip.mp4")


if __name__ == "__main__":
    unittest.main()
