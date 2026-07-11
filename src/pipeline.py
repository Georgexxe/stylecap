"""Reusable StyleCap pipeline shared by the evaluator, CLI, and demo app."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import compact, config, ingest, perceive
from .evaluator import EvaluationResult, EvaluationTask
from .schemas import FactSheet, FinalCaption


@dataclass(frozen=True)
class PipelineResult:
    """Grounded intermediate facts and selected final captions for one clip."""

    facts: FactSheet
    captions: list[FinalCaption]


def process_clip(
    clip_path: str,
    styles: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the complete pipeline for one local video clip."""
    if progress:
        progress("sample")
    media = ingest.run(clip_path)
    if progress:
        progress("perceive")
    facts = perceive.run(media)
    if progress:
        progress("style")
    captions = compact.run(facts, styles, frame_paths=media["frames"])
    if progress:
        progress("select")
    return PipelineResult(facts=facts, captions=captions)


def process_task(task: EvaluationTask) -> EvaluationResult:
    """Download and process one official evaluator task."""
    video_path = ingest.download_video(
        task.video_url,
        Path(config.CACHE_DIR) / "downloads",
    )
    result = process_clip(str(video_path), styles=task.styles)
    return EvaluationResult(
        task_id=task.task_id,
        captions={caption.style: caption.caption for caption in result.captions},
    )
