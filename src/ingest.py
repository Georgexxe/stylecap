"""Download videos and create cached frame and transcript representations."""

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TypedDict, cast

import httpx

from . import config

MAX_VIDEO_BYTES = 512 * 1024 * 1024


class TranscriptItem(TypedDict):
    """One timestamped ASR segment."""

    t: float
    text: str


class Media(TypedDict):
    """Cached media representation consumed by the perception stage."""

    clip_id: str
    clip_hash: str
    frames: list[str]
    transcript: list[TranscriptItem]


class DownloadError(RuntimeError):
    """Raised when an evaluator clip cannot be downloaded safely."""


def download_video(url: str, cache_dir: Path, max_bytes: int = MAX_VIDEO_BYTES) -> Path:
    """Download a remote evaluation clip once and return its cached path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    destination = cache_dir / f"{digest}.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    temporary = destination.with_suffix(".part")
    downloaded = 0
    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=httpx.Timeout(90.0, connect=15.0),
        ) as response:
            response.raise_for_status()
            declared_size = int(response.headers.get("content-length", "0"))
            if declared_size > max_bytes:
                raise DownloadError(
                    f"video exceeds {max_bytes} byte download limit: {declared_size}"
                )
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise DownloadError(f"video exceeded {max_bytes} byte download limit")
                    output.write(chunk)
        if downloaded == 0:
            raise DownloadError("downloaded video is empty")
        temporary.replace(destination)
        return destination
    except (httpx.HTTPError, OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise DownloadError(f"could not download {url}: {exc}") from exc


def _clip_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_dir(clip_hash: str) -> str:
    d = os.path.join(config.CACHE_DIR, clip_hash)
    os.makedirs(d, exist_ok=True)
    return d


def _extract_frames(clip: str, out_dir: str) -> list[str]:
    """Extract deterministic temporal samples capped at ``MAX_FRAMES``."""
    for stale_frame in Path(out_dir).glob("*.jpg"):
        stale_frame.unlink()
    uni = os.path.join(out_dir, "u%03d.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            clip,
            "-vf",
            f"fps={config.TARGET_FPS},scale=640:-1",
            "-q:v",
            "4",
            uni,
        ],
        check=True,
    )
    frames = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))
    if len(frames) > config.MAX_FRAMES:
        step = len(frames) / config.MAX_FRAMES
        frames = [frames[int(i * step)] for i in range(config.MAX_FRAMES)]
    return [os.path.join(out_dir, f) for f in frames]


def _transcribe(clip: str) -> list[TranscriptItem]:
    """ASR via faster-whisper; returns [{t, text}]. Empty list if no/unusable audio."""
    if not config.ENABLE_ASR:
        return []
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        model = WhisperModel(config.WHISPER_MODEL, compute_type="int8")
        segments, _info = model.transcribe(clip, vad_filter=True)
        return [{"t": round(s.start, 1), "text": s.text.strip()} for s in segments]
    except Exception as e:
        print(f"  ASR skipped ({e.__class__.__name__}: {e})")
        return []


def frame_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def run(clip_path: str) -> Media:
    """Returns {clip_id, clip_hash, frames: [paths], transcript: [{t, text}]}, cached."""
    ch = _clip_hash(clip_path)
    cdir = _cache_dir(ch)
    meta_path = os.path.join(cdir, "ingest.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return cast(Media, json.load(f))

    frames_dir = os.path.join(cdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    meta: Media = {
        "clip_id": os.path.splitext(os.path.basename(clip_path))[0],
        "clip_hash": ch,
        "frames": _extract_frames(clip_path, frames_dir),
        "transcript": _transcribe(clip_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
