"""Low-latency generation and selection path for Track 2 evaluation."""

from typing import Any, cast

from . import config, llm, stylize
from .schemas import FactSheet, FinalCaption

CANDIDATES_PER_STYLE = 3

GENERATION_SYSTEM = """You are a grounded video-caption writer. Generate exactly three
distinct candidates for every requested style. Every factual detail must be supported by
the supplied fact sheet. Humor may change the framing, never the observable events.
Do not mention incidental signs or on-screen text unless they are central to the main action.
Return only JSON in this shape:
{"candidates": {"<style>": ["candidate 0", "candidate 1", "candidate 2"]}}.

STYLE RULES:
{style_rules}"""

SELECTION_SYSTEM = """You are a strict video-caption evaluator. For each requested style,
select the single candidate that best balances factual accuracy and unmistakable style
match. Reject invented people, objects, actions, dialogue, quantities, or outcomes.
Prefer candidates focused on the main visible action; reject irrelevant signage and incidental
on-screen text.
Return only JSON in this shape:
{"selected": {"<style>": 0}}.
The value for each style must be the zero-based index of one supplied candidate."""


class CompactCaptionError(RuntimeError):
    """Raised when batched generation or selection returns an invalid result."""


def _validate_candidates(data: dict[str, Any], styles: list[str]) -> dict[str, list[str]]:
    candidates = data.get("candidates")
    if not isinstance(candidates, dict):
        raise CompactCaptionError("generation response has no candidates object")
    if set(candidates) != set(styles):
        raise CompactCaptionError("generation response does not match requested styles")
    validated: dict[str, list[str]] = {}
    for style in styles:
        values = candidates.get(style)
        if not isinstance(values, list) or len(values) != CANDIDATES_PER_STYLE:
            raise CompactCaptionError(
                f"{style}: expected exactly {CANDIDATES_PER_STYLE} candidates"
            )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise CompactCaptionError(f"{style}: candidates must be non-empty strings")
        validated[style] = cast(list[str], values)
    return validated


def run(facts: FactSheet, styles: list[str] | None = None) -> list[FinalCaption]:
    """Generate and select all requested captions using exactly two model calls."""
    requested = list(styles or config.STYLES)
    unknown = set(requested) - set(config.STYLES)
    if unknown:
        raise CompactCaptionError(f"unknown requested styles: {sorted(unknown)}")
    style_rules = "\n\n".join(f"## {style}\n{stylize._load_spec(style)}" for style in requested)
    generated = llm.chat_json(
        config.STYLE_MODEL,
        [
            {
                "role": "system",
                "content": GENERATION_SYSTEM.replace("{style_rules}", style_rules),
            },
            {
                "role": "user",
                "content": (
                    f"Fact sheet:\n{facts.model_dump_json(indent=2)}\n\n"
                    f"Requested styles: {requested}"
                ),
            },
        ],
        stage="caption_batch",
        temperature=0.75,
        max_tokens=1200,
    )
    candidates = _validate_candidates(generated, requested)

    judge_model = config.JUDGE_MODELS[0] if config.JUDGE_MODELS else config.STYLE_MODEL
    selected_data = llm.chat_json(
        judge_model,
        [
            {"role": "system", "content": SELECTION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Fact sheet:\n{facts.model_dump_json(indent=2)}\n\nCandidates:\n{generated}"
                ),
            },
        ],
        stage="select_batch",
        temperature=0.0,
        max_tokens=300,
    )
    selected = selected_data.get("selected")
    if not isinstance(selected, dict) or set(selected) != set(requested):
        raise CompactCaptionError("selection response does not match requested styles")

    results = []
    for style in requested:
        index = selected.get(style)
        if not isinstance(index, int) or not 0 <= index < CANDIDATES_PER_STYLE:
            raise CompactCaptionError(f"{style}: selected index is out of range")
        results.append(
            FinalCaption(
                clip_id=facts.clip_id,
                style=style,
                caption=candidates[style][index].strip(),
            )
        )
    return results
