"""High-recall single-call captioning path used by the Track 2 evaluator."""

import json
import re
from typing import Any

from . import config, ingest, llm
from .schemas import FinalCaption

MIN_WORDS = 24
MAX_WORDS = 70

STYLE_GUIDANCE = {
    "formal": (
        "Professional, objective, and precise. Describe the setting, important subjects, "
        "actions, and visible changes in neutral third-person language."
    ),
    "sarcastic": (
        "Dry, unmistakably ironic, and lightly mocking while still accurately describing "
        "the actual subjects and events in the clip."
    ),
    "humorous_tech": (
        "Clearly funny for a technical audience, using a specific software, hardware, API, "
        "debugging, or engineering analogy grounded in the visible action."
    ),
    "humorous_non_tech": (
        "Clearly funny through everyday, broadly relatable observations with no technical "
        "jargon, while retaining the clip's concrete subjects and events."
    ),
}

SYSTEM_PROMPT = """You are an expert video captioner evaluated by an LLM judge for both
factual accuracy and tone. You will receive evenly sampled frames from one continuous video
in chronological order. Generate one rich caption for every requested style.

Each caption must:
- contain 2-3 complete sentences and roughly 30-55 words;
- cover the main setting, subjects, actions, and meaningful changes across time;
- remain faithful to visible evidence while being unmistakably written in its assigned tone;
- use concrete nouns and verbs instead of vague summaries;
- stand alone without labels, hashtags, preambles, or references to these instructions.

Style guidance:
{style_guidance}

Return only a JSON object whose keys exactly match the requested styles and whose values are
caption strings. Do not use markdown fences or add any text outside the JSON object."""


class ScoringCaptionError(RuntimeError):
    """Raised when the direct scoring model returns unusable captions."""


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _validate(payload: dict[str, Any], styles: list[str]) -> dict[str, str]:
    if set(payload) != set(styles):
        raise ScoringCaptionError("response keys do not match requested styles")

    captions: dict[str, str] = {}
    for style in styles:
        value = payload.get(style)
        if not isinstance(value, str) or not value.strip():
            raise ScoringCaptionError(f"{style}: caption must be a non-empty string")
        caption = " ".join(value.split())
        word_count = len(caption.split())
        if not MIN_WORDS <= word_count <= MAX_WORDS:
            raise ScoringCaptionError(
                f"{style}: expected {MIN_WORDS}-{MAX_WORDS} words, got {word_count}"
            )
        captions[style] = caption

    for index, style in enumerate(styles):
        left = _normalized_words(captions[style])
        for other_style in styles[index + 1 :]:
            right = _normalized_words(captions[other_style])
            overlap = len(left & right) / max(1, min(len(left), len(right)))
            if overlap > 0.88:
                raise ScoringCaptionError(
                    f"{style} and {other_style}: captions are insufficiently distinct"
                )
    return captions


def run(frame_paths: list[str], styles: list[str]) -> list[FinalCaption]:
    """Generate all requested styles directly from temporal visual evidence."""
    unknown = set(styles) - set(config.STYLES)
    if unknown:
        raise ScoringCaptionError(f"unknown requested styles: {sorted(unknown)}")
    if not frame_paths:
        raise ScoringCaptionError("no video frames were extracted")

    style_guidance = "\n".join(
        f'- {style}: {STYLE_GUIDANCE[style]}' for style in styles
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Requested styles: {json.dumps(styles)}\n"
                f"The following {len(frame_paths)} frames span the full clip."
            ),
        }
    ]
    for frame_path in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": ingest.frame_to_data_url(frame_path)},
            }
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.replace("{style_guidance}", style_guidance),
        },
        {"role": "user", "content": content},
    ]
    last_error: Exception | None = None
    for attempt, temperature in enumerate((0.45, 0.35), start=1):
        try:
            payload = llm.chat_json(
                config.SCORING_MODEL,
                messages,
                stage="scoring_caption",
                temperature=temperature,
                max_tokens=1100,
            )
            captions = _validate(payload, styles)
            return [
                FinalCaption(clip_id="evaluation", style=style, caption=captions[style])
                for style in styles
            ]
        except (ScoringCaptionError, ValueError) as exc:
            last_error = exc
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Attempt {attempt} failed validation: {exc}. Regenerate the complete "
                        "JSON object and satisfy every length, accuracy, and tone requirement."
                    ),
                }
            )
    raise ScoringCaptionError(f"caption generation failed validation: {last_error}")
