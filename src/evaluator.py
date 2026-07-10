"""Official Track 2 file contract and evaluation orchestration."""

import json
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from . import config


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
    """Process every task and atomically write exact Track 2 results JSON."""
    tasks = load_tasks(input_path)
    results = []
    for task in tasks:
        result = processor(task)
        result.validate_for(task)
        results.append(result)

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
    return results
