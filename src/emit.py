"""Validate and export local per-caption JSONL files.

The official Track 2 evaluator contract is implemented in ``src.evaluator``. This module
supports the local directory-processing and demonstration commands.
"""

import os

from . import config
from .schemas import FinalCaption


class FormatError(ValueError):
    pass


def _validate(captions: list[FinalCaption], expected_clip_ids: set[str] | None = None) -> None:
    by_clip: dict[str, set[str]] = {}
    for c in captions:
        if c.style not in config.STYLES:
            raise FormatError(f"unknown style {c.style!r} for clip {c.clip_id}")
        if not c.caption or not c.caption.strip():
            raise FormatError(f"empty caption: {c.clip_id}/{c.style}")
        low = c.caption.lower()
        for meta in ("here's a", "here is a", "caption:", "as an ai"):
            if meta in low:
                raise FormatError(f"meta-text leaked in {c.clip_id}/{c.style}: {c.caption[:60]!r}")
        styles = by_clip.setdefault(c.clip_id, set())
        if c.style in styles:
            raise FormatError(f"duplicate style {c.style!r} for clip {c.clip_id}")
        styles.add(c.style)

    if expected_clip_ids is not None:
        unexpected = set(by_clip) - expected_clip_ids
        if unexpected:
            raise FormatError(f"unexpected clips: {sorted(unexpected)}")
        missing_clips = expected_clip_ids - set(by_clip)
        if missing_clips:
            raise FormatError(f"missing clips: {sorted(missing_clips)}")
    for clip_id, styles in by_clip.items():
        missing = set(config.STYLES) - styles
        if missing:
            raise FormatError(f"clip {clip_id} missing styles: {sorted(missing)}")


def write(
    captions: list[FinalCaption],
    out_dir: str,
    expected_clip_ids: set[str] | None = None,
) -> str:
    """Validate and write captions in the configured submission format."""
    _validate(captions, expected_clip_ids)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "captions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in captions:
            f.write(c.model_dump_json() + "\n")
    return path
