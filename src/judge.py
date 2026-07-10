"""Optional high-depth caption evaluation and regeneration pipeline.

Candidates pass factual verification before tone scoring to prioritize grounded output.
"""

import random

from . import config, llm, stylize
from .schemas import Candidate, FactSheet, FinalCaption

ACCURACY_SYS = """You are a fact-checker. Given a fact sheet and a caption, extract the
caption's factual claims (ignore opinions, irony framing, and stylistic exaggeration of
REAL events — 'files it under gravity research' about a real falling glass is fine;
a glass that never fell is not). Return JSON:
{"claims": [...], "unverified": [claims with NO support in the fact sheet]}"""

TONE_SYS = """You are a strict caption judge. Score the caption 1-10 on each dimension
and return JSON {"accuracy": n, "tone": n, "specificity": n, "critique": "one sentence"}.
- accuracy: consistent with the fact sheet, no invented details
- tone: unmistakably matches the target style '{style}' (committed, not half-hearted)
- specificity: cites concrete, distinctive details from this exact video"""

CLASSIFY_SYS = """Classify the caption's style. Options: formal, sarcastic,
humorous_tech, humorous_non_tech. Return JSON {"style": "<option>"}."""


class CaptionSelectionError(RuntimeError):
    """Raised when no grounded candidate can be selected for every style."""


def accuracy_gate(facts: FactSheet, cand: Candidate) -> Candidate:
    res = llm.chat_json(
        config.JUDGE_MODELS[0],
        [
            {"role": "system", "content": ACCURACY_SYS},
            {
                "role": "user",
                "content": (
                    f"Fact sheet:\n{facts.model_dump_json()}\n\n"
                    f"Caption ({cand.style}): {cand.caption}\n\n"
                    "Extract claims and list unverified ones."
                ),
            },
        ],
        stage="judge",
        temperature=0.0,
    )
    cand.unverified_claims = res.get("unverified", [])
    cand.accuracy_pass = len(cand.unverified_claims) == 0
    return cand


def tone_scores(facts: FactSheet, cand: Candidate) -> Candidate:
    for jm in config.JUDGE_MODELS:
        res = llm.chat_json(
            jm,
            [
                {"role": "system", "content": TONE_SYS.replace("{style}", cand.style)},
                {
                    "role": "user",
                    "content": (
                        f"Fact sheet:\n{facts.model_dump_json()}\n\n"
                        f"Target style: {cand.style}\n"
                        f"Caption: {cand.caption}\n\nScore it."
                    ),
                },
            ],
            stage="judge",
            temperature=0.0,
        )
        cand.tone_scores[jm] = (res["accuracy"] + res["tone"] + res["specificity"]) / 3
        cand.critique = res.get("critique")
    cand.mean_score = sum(cand.tone_scores.values()) / len(cand.tone_scores)
    return cand


def contrast_ok(winners: dict[str, Candidate]) -> list[str]:
    """Blind style classification; returns styles that were misidentified."""
    confused = []
    items = list(winners.items())
    random.shuffle(items)
    for style, cand in items:
        res = llm.chat_json(
            config.JUDGE_MODELS[0],
            [
                {"role": "system", "content": CLASSIFY_SYS},
                {"role": "user", "content": f"Classify the caption. Caption: {cand.caption}"},
            ],
            stage="judge",
            temperature=0.0,
        )
        if res.get("style") != style:
            confused.append(style)
    return confused


def run(facts: FactSheet, candidates: list[Candidate]) -> list[FinalCaption]:
    winners: dict[str, Candidate] = {}
    for style in config.STYLES:
        pool = [c for c in candidates if c.style == style]
        for round_no in range(config.MAX_REGEN_ROUNDS + 1):
            pool = [accuracy_gate(facts, c) for c in pool]
            alive = [c for c in pool if c.accuracy_pass]
            if not alive:  # every candidate hallucinated -> regenerate stricter
                pool = stylize.run(
                    facts,
                    [style],
                    extra_instruction="Previous attempts invented details. "
                    "Stick strictly to fact-sheet facts.",
                )
                continue
            alive = [tone_scores(facts, c) for c in alive]
            best = max(alive, key=lambda c: c.mean_score or 0)
            if (best.mean_score or 0) >= 7.0 or round_no == config.MAX_REGEN_ROUNDS:
                winners[style] = best
                break
            pool = stylize.run(
                facts,
                [style],
                extra_instruction=f"A judge critiqued the last attempt: "
                f"'{best.critique}'. Fix that specifically.",
            )

        if style not in winners:
            raise CaptionSelectionError(
                f"clip {facts.clip_id!r}: no grounded candidate survived for {style!r}"
            )

    for style in contrast_ok(winners):
        others = "; ".join(f"{s}: {c.caption}" for s, c in winners.items() if s != style)
        regen = stylize.run(
            facts,
            [style],
            k=2,
            extra_instruction=f"Make this caption UNMISTAKABLY '{style}' and "
            f"clearly distinct from these other captions: {others}",
        )
        regen = [accuracy_gate(facts, c) for c in regen]
        regen = [tone_scores(facts, c) for c in regen if c.accuracy_pass]
        if regen:
            winners[style] = max(regen, key=lambda c: c.mean_score or 0)

    return [
        FinalCaption(clip_id=facts.clip_id, style=s, caption=c.caption) for s, c in winners.items()
    ]
