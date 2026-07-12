"""Official Track 2 file contract and evaluation orchestration."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from . import config

LOGGER = logging.getLogger(__name__)

FALLBACK_CAPTIONS = {
    "formal": "The video presents a short sequence of visible activity.",
    "sarcastic": "A brief scene unfolds, naturally keeping its finer details mysterious.",
    "humorous_tech": (
        "The clip runs its visual workflow while the finer telemetry stays unavailable."
    ),
    "humorous_non_tech": "A short scene plays out and keeps the smallest details to itself.",
}


class ContractError(ValueError):
    """Raised when evaluator input or output violates the published schema."""


class EvaluationTask(BaseModel):
    """One video-captioning task from ``/input/tasks.json``."""

    task_id: str = Field(min_length=1)
    video_url: str
    styles: list[str] = Field(min_length=1)

    @field_validator("video_url")
    @classmethod
    def validate_video_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("video_url must be an http(s) URL")
        return value

    @field_validator("styles")
    @classmethod
    def validate_styles(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("styles must not contain duplicates")
        unknown = set(value) - set(config.STYLES)
        if unknown:
            raise ValueError(f"unknown styles: {sorted(unknown)}")
        return value


class EvaluationResult(BaseModel):
    """One task result written to ``/output/results.json``."""

    task_id: str = Field(min_length=1)
    captions: dict[str, str]

    def validate_for(self, task: EvaluationTask) -> None:
        """Check task identity and exact requested-style completeness."""
        if self.task_id != task.task_id:
            raise ContractError(
                f"task ID mismatch: expected {task.task_id!r}, got {self.task_id!r}"
            )
        missing = set(task.styles) - set(self.captions)
        if missing:
            raise ContractError(f"missing requested styles: {sorted(missing)}")
        unexpected = set(self.captions) - set(task.styles)
        if unexpected:
            raise ContractError(f"unexpected styles: {sorted(unexpected)}")
        empty = [style for style, caption in self.captions.items() if not caption.strip()]
        if empty:
            raise ContractError(f"empty captions: {sorted(empty)}")


TaskProcessor = Callable[[EvaluationTask], EvaluationResult]


def fallback_result(task: EvaluationTask) -> EvaluationResult:
    """Build a schema-complete last-resort result for one evaluator task."""
    return EvaluationResult(
        task_id=task.task_id,
        captions={style: FALLBACK_CAPTIONS[style] for style in task.styles},
    )


def write_results(output_path: Path, results: list[EvaluationResult]) -> None:
    """Atomically publish results so the evaluator never observes partial JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            [result.model_dump() for result in results],
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def load_tasks(path: Path) -> list[EvaluationTask]:
    """Load and validate all evaluator tasks."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid task JSON from {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ContractError("tasks.json must contain a non-empty JSON array")
    tasks = [EvaluationTask.model_validate(item) for item in payload]
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ContractError("tasks.json contains duplicate task_id values")
    return tasks


def run_evaluation(
    input_path: Path,
    output_path: Path,
    *,
    processor: TaskProcessor,
) -> list[EvaluationResult]:
    """Process tasks while always preserving a complete Track 2 results file."""
    tasks = load_tasks(input_path)
    results = [fallback_result(task) for task in tasks]
    write_results(output_path, results)

    for index, task in enumerate(tasks):
        try:
            result = processor(task)
            result.validate_for(task)
        except Exception as exc:
            LOGGER.error(
                "task %s failed; retaining schema-safe fallback: %s",
                task.task_id,
                exc,
            )
            continue
        results[index] = result
        write_results(output_path, results)
    return results
