"""Create a cached, grounded fact sheet from sampled frames and a transcript."""

import hashlib
import json
import os
from typing import Any

from . import config, ingest, llm
from .schemas import FactSheet

SYSTEM = """You are a meticulous video analyst. You will receive sampled frames from a
short video (in chronological order) and an audio transcript. Produce a fact sheet as a
single JSON object. Report ONLY what is directly observable — no guesses, no
embellishment. Accuracy is critical: downstream writers may only use facts
you record.

JSON keys:
Text rule: only record on-screen text that is clearly legible in at least one frame.
Omit uncertain, partially obscured, or guessed text.
- clip_id (string, echo back)
- setting (string)
- entities (list of strings — people/animals/objects that matter)
- actions (list of strings — what actually happens, concrete verbs)
- timeline (list of {t, event} — 3-8 key beats with approx timestamps like "0:41")
- on_screen_text (list of strings — EXACT text visible in frames, verbatim)
- audio_events (list of strings — notable sounds and short VERBATIM quotes from transcript)
- mood (string)
- humor_hooks (list of {observation, why_funny} — 3-5 incongruities, subverted
  expectations, or relatable beats a comedy writer could use; each must reference
  something actually visible/audible)"""


def _cache_path(media: ingest.Media) -> str:
    """Keep mock output and model-specific perception results isolated."""
    identity = f"{config.PERCEPTION_MODEL}|mock={int(llm.MOCK)}"
    model_key = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return os.path.join(config.CACHE_DIR, media["clip_hash"], f"factsheet-{model_key}.json")


def run(media: ingest.Media) -> FactSheet:
    cache_path = _cache_path(media)
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return FactSheet.model_validate_json(f.read())

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"clip_id: {media['clip_id']}\n"
            f"Transcript (may be empty): {json.dumps(media['transcript'])}\n"
            f"Now the {len(media['frames'])} frames, chronological. "
            f"Return the fact sheet JSON.",
        }
    ]
    for fp in media["frames"]:
        content.append({"type": "image_url", "image_url": {"url": ingest.frame_to_data_url(fp)}})

    data = llm.chat_json(
        config.PERCEPTION_MODEL,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        stage="perceive",
        temperature=0.2,
        max_tokens=2048,
    )
    data["clip_id"] = media["clip_id"]  # trust our id, not the model's echo
    facts = FactSheet.model_validate(data)

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(facts.model_dump_json(indent=2))
    return facts
