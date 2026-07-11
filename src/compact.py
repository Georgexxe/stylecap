"""Low-latency, accuracy-first generation path for Track 2 evaluation."""

import json
from typing import Any, cast

from . import config, ingest, llm
from .schemas import FactSheet, FinalCaption

CANDIDATES_PER_STYLE = 4
MAX_SELECTION_FRAMES = 12

SCORING_STYLE_RULES = """formal:
- Write 18-38 words in neutral broadcast language.
- State the visible subject, main action, and one useful setting detail literally.
- No humor, opinion, inferred emotion, or metaphor.

sarcastic:
- Write 18-42 words with unmistakable dry irony and a clean final turn.
- Keep the visible subject and action explicit; add no motive, backstory, dialogue, or outcome.
- Aim the irony at a specific visible detail or contrast.
- Avoid generic filler such as "truly thrilling" or "a groundbreaking event."

humorous_tech:
- Write 18-42 words with one familiar software metaphor and a clear punchline.
- State the visible subject and action explicitly before or inside the metaphor.
- Do not invent an error message, interface, system failure, user intent, or unseen result.
- Tie the software metaphor to the clip's exact visible action, not merely its subject.

humorous_non_tech:
- Write 18-42 words with a warm, unmistakable observational punchline.
- State the visible subject and action explicitly.
- Build the punchline from a specific visible action or contrast; avoid generic "chaos."
- Use everyday language only; no software, business, or internet jargon.
- Do not invent a destination, relationship, thought, plan, dialogue, or backstory."""

GENERATION_SYSTEM = """You write captions for an automated video-grounding evaluator.
Generate exactly four distinct candidates for every requested style.

ACCURACY IS THE PRIMARY SCORE:
- Every concrete claim must be directly supported by the supplied observable facts.
- Preserve the main visible subject, action, and setting in every candidate.
- Never infer intent, thoughts, emotions, identity, relationships, causes, destinations,
  dialogue, quantities, or events outside the sampled video.
- A metaphor may frame a visible event but may not introduce another event as if it happened.
- Do not mention incidental signs or text unless central to the main visible action.
- Prefer precise, plain wording over clever but weakly grounded wording.
- Do not use labels, preambles, hashtags, emojis, or quotation marks around the whole caption.

FOR EACH STYLE, MAKE THE FOUR OPTIONS PURPOSEFULLY DIFFERENT:
0. Literal-safe: maximum factual clarity with restrained style.
1. Balanced: strong style while preserving every central fact.
2. Detail-led: build the tone around the most distinctive visible action or contrast.
3. Punchline-led: the boldest grounded version, with no unsupported claim.
Do not repeat the same sentence with small word substitutions.

Return only JSON in this shape:
{"candidates": {"<style>": ["candidate 0", "candidate 1", "candidate 2", "candidate 3"]}}.

STYLE RULES:
{style_rules}"""

SELECTION_SYSTEM = """You are a strict proxy for an automated video-caption scorer.
For each requested style, select one candidate using this priority order:
1. Direct factual support for every claim; any unsupported assumption is disqualifying.
2. Coverage of the main visible subject and action.
3. Clear match to the requested style.
4. Specificity: reward wording that depends on a distinctive detail from this exact clip.
5. Concision and natural wording.

Reject invented intent, thoughts, emotions, identity, relationships, backstory, dialogue,
causes, destinations, quantities, error messages, interface behavior, or unseen outcomes.
Reject captions whose joke replaces the actual video description. Prefer a slightly simpler
caption over a cleverer caption with weaker grounding. Among equally grounded candidates,
reject generic jokes that could be pasted onto an unrelated video and choose the candidate
whose tone depends on a concrete detail from this clip.
Formal must remain entirely literal. Sarcastic must read as dry irony. Humorous-tech must
contain a coherent software metaphor. Humorous-non-tech must be clearly funny in everyday
language with no technical jargon. Do not choose a safe candidate when an equally grounded
candidate has a clearer style match.
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


def _selection_frames(frame_paths: list[str]) -> list[str]:
    """Choose a small chronological evidence set spanning the entire clip."""
    if len(frame_paths) <= MAX_SELECTION_FRAMES:
        return frame_paths
    last = len(frame_paths) - 1
    indices = [
        round(index * last / (MAX_SELECTION_FRAMES - 1))
        for index in range(MAX_SELECTION_FRAMES)
    ]
    return [frame_paths[index] for index in dict.fromkeys(indices)]


def _scoring_facts(facts: FactSheet) -> dict[str, Any]:
    """Remove subjective metadata and unsupported model-invented timestamps."""
    data = facts.model_dump(exclude={"humor_hooks", "mood"})
    data["timeline_events"] = [beat["event"] for beat in data.pop("timeline", [])]
    return data


def run(
    facts: FactSheet,
    styles: list[str] | None = None,
    *,
    frame_paths: list[str] | None = None,
) -> list[FinalCaption]:
    """Generate and select all requested captions using exactly two model calls."""
    requested = list(styles or config.STYLES)
    unknown = set(requested) - set(config.STYLES)
    if unknown:
        raise CompactCaptionError(f"unknown requested styles: {sorted(unknown)}")
    scoring_facts = _scoring_facts(facts)
    generated = llm.chat_json(
        config.STYLE_MODEL,
        [
            {
                "role": "system",
                "content": GENERATION_SYSTEM.replace("{style_rules}", SCORING_STYLE_RULES),
            },
            {
                "role": "user",
                "content": (
                    f"Observable facts:\n{json.dumps(scoring_facts, indent=2)}\n\n"
                    f"Requested styles: {requested}"
                ),
            },
        ],
        stage="caption_batch",
        temperature=0.55,
        max_tokens=1200,
    )
    candidates = _validate_candidates(generated, requested)

    selection_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Observable facts:\n{json.dumps(scoring_facts, indent=2)}\n\n"
                f"Candidates:\n{json.dumps(candidates, indent=2)}\n\n"
                "Representative frames follow in chronological order. Use them as the "
                "final authority when a candidate and the fact sheet disagree."
            ),
        }
    ]
    evidence_frames = _selection_frames(frame_paths or [])
    for frame_number, frame_path in enumerate(evidence_frames, start=1):
        selection_content.append(
            {
                "type": "text",
                "text": f"Evidence frame {frame_number} of {len(evidence_frames)}:",
            }
        )
        selection_content.append(
            {
                "type": "image_url",
                "image_url": {"url": ingest.frame_to_data_url(frame_path)},
            }
        )

    judge_model = config.JUDGE_MODELS[0] if config.JUDGE_MODELS else config.STYLE_MODEL
    selected_data = llm.chat_json(
        judge_model,
        [
            {"role": "system", "content": SELECTION_SYSTEM},
            {"role": "user", "content": selection_content},
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
        selected_index = selected.get(style)
        if not isinstance(selected_index, int) or not 0 <= selected_index < CANDIDATES_PER_STYLE:
            raise CompactCaptionError(f"{style}: selected index is out of range")
        results.append(
            FinalCaption(
                clip_id=facts.clip_id,
                style=style,
                caption=candidates[style][selected_index].strip(),
            )
        )
    return results
