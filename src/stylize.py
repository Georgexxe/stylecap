"""Stage 3: FactSheet -> k caption candidates per style, using styles/*.md specs."""

import os

from . import config, llm
from .schemas import Candidate, FactSheet

STYLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles")

SYSTEM_TMPL = """You write video captions in ONE specific style. Follow the style spec
below exactly — voice, rules, banned moves, and the register of its reference examples.

HARD CONSTRAINT: you may only reference facts present in the fact sheet you are given.
Do not invent objects, actions, dialogue, or details. Creativity lives in the ANGLE,
never in fabricated content. Output ONLY the caption text — no labels, no preamble,
no explanation of the joke.

STYLE SPEC:
{spec}"""


def _load_spec(style: str) -> str:
    filename = style.replace("_", "-")
    with open(os.path.join(STYLES_DIR, f"{filename}.md"), encoding="utf-8") as f:
        return f.read()


def run(
    facts: FactSheet,
    styles: list[str] | None = None,
    k: int | None = None,
    extra_instruction: str = "",
) -> list[Candidate]:
    styles = styles or config.STYLES
    k = k or config.K_CANDIDATES
    out: list[Candidate] = []
    for style in styles:
        spec = _load_spec(style)
        user = (
            f"Fact sheet:\n{facts.model_dump_json(indent=2)}\n\n"
            f"Write ONE caption in the '{style}' style. 1-3 sentences."
            + (f"\n\nAdditional instruction: {extra_instruction}" if extra_instruction else "")
        )
        for _ in range(k):
            text = (
                llm.chat(
                    config.STYLE_MODEL,
                    [
                        {"role": "system", "content": SYSTEM_TMPL.format(spec=spec)},
                        {"role": "user", "content": user},
                    ],
                    stage="stylize",
                    temperature=config.TEMPS.get(style, 0.8),
                    max_tokens=200,
                )
                .strip()
                .strip('"')
            )
            out.append(Candidate(clip_id=facts.clip_id, style=style, caption=text))
    return out
