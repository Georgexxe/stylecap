"""Fireworks gateway with retries, usage tracking, JSON handling, and mock mode.

MOCK mode (STYLECAP_MOCK=1) returns deterministic fixture responses so the entire
pipeline can be verified offline without API usage.
"""

import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, wait_fixed

from . import budget, config

MOCK = os.environ.get("STYLECAP_MOCK", "0") == "1"

Message = dict[str, Any]

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.FIREWORKS_BASE_URL,
            api_key=os.environ["FIREWORKS_API_KEY"],
            max_retries=0,
            timeout=45.0,
        )
    return _client


def _mock_response(messages: list[Message], want_json: bool) -> str:
    """Deterministic fake keyed on the prompt hash — stable across runs."""
    seed = hashlib.sha256(json.dumps(messages, default=str).encode()).hexdigest()[:8]
    text = " ".join(str(m.get("content", ""))[:2000] for m in messages).lower()
    if want_json and "expert video captioner" in text and "2-3 complete sentences" in text:
        styles = [style for style in config.STYLES if style in text] or config.STYLES
        options = {
            "formal": (
                "An orange cat crosses a bright kitchen counter while watching the camera "
                "and pushes a glass toward the edge. The glass drops out of view, leaving "
                "the animal standing beside the remaining objects on the work surface."
            ),
            "sarcastic": (
                "An orange cat conducts a very serious inspection of the kitchen counter "
                "before casually sending a glass over the edge. Apparently gravity needed "
                "another thorough demonstration, and the cat was uniquely qualified to provide it."
            ),
            "humorous_tech": (
                "The orange cat runs a countertop stress test, selects one glass, and executes "
                "a manual drop command while staring into the camera. Gravity accepts the "
                "request immediately, proving this particular API has no confirmation dialog."
            ),
            "humorous_non_tech": (
                "An orange cat strolls along the kitchen counter and gives one glass the tiny "
                "push it never asked for. The glass disappears over the side while its furry "
                "coworker remains behind, looking remarkably pleased with the day's productivity."
            ),
        }
        return json.dumps({style: options[style] for style in styles})
    # Several prompts contain "fact sheet", so match specific intents before perception.
    if want_json and "exactly four" in text and "candidates" in text:
        styles = [style for style in config.STYLES if style in text] or config.STYLES
        return json.dumps(
            {
                "candidates": {
                    style: [f"Mock {style} caption candidate {index}." for index in range(4)]
                    for style in styles
                }
            }
        )
    if want_json and "zero-based index" in text:
        styles = [style for style in config.STYLES if style in text] or config.STYLES
        return json.dumps({"selected": {style: 0 for style in styles}})
    if want_json and "unverified" in text:
        return json.dumps({"claims": ["cat knocks glass off counter"], "unverified": []})
    if want_json and "score" in text:
        # vary score deterministically by seed so best-of-n has a real argmax
        s = 6 + int(seed, 16) % 4
        return json.dumps(
            {"accuracy": s, "tone": s - 1, "specificity": s, "critique": f"mock critique {seed}"}
        )
    if want_json and "classify" in text:
        for st in ("humorous_non_tech", "humorous_tech", "sarcastic", "formal"):
            if st in text:
                return json.dumps({"style": st})
        return json.dumps({"style": "sarcastic"})
    if want_json and "fact sheet" in text:
        return json.dumps(
            {
                "clip_id": f"mock-{seed}",
                "setting": "a kitchen counter",
                "entities": ["a cat", "three glasses"],
                "actions": ["cat walks along counter", "cat knocks third glass off"],
                "timeline": [
                    {"t": "0:05", "event": "cat steps onto counter"},
                    {"t": "0:32", "event": "third glass falls"},
                ],
                "on_screen_text": [],
                "audio_events": ["off-screen voice: 'Whiskers, no'"],
                "mood": "chaotic-domestic",
                "humor_hooks": [
                    {
                        "observation": "cat stares at camera while pushing glass",
                        "why_funny": "premeditation + eye contact",
                    }
                ],
            }
        )
    return (
        f"[mock caption {seed}] The cat, with full eye contact, "
        "files glass number three under 'gravity research'."
    )


def _is_retryable(exc: BaseException) -> bool:
    """Retry only failures that Fireworks documents as transient."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 429, 502, 503, 504, 520}
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=2, max=12),
    reraise=True,
)
def _call(
    model: str,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
    want_json: bool,
) -> tuple[str, int, int]:
    typed_messages = cast(Iterable[ChatCompletionMessageParam], messages)
    if want_json:
        response_format: ResponseFormatJSONObject = {"type": "json_object"}
        resp = _get_client().chat.completions.create(
            model=model,
            messages=typed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body={"reasoning_effort": "none"},
        )
    else:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=typed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"reasoning_effort": "none"},
        )
    u = resp.usage
    return (
        resp.choices[0].message.content or "",
        (u.prompt_tokens if u else 0),
        (u.completion_tokens if u else 0),
    )


def chat(
    model: str,
    messages: list[Message],
    stage: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    want_json: bool = False,
) -> str:
    """All stages call this. Returns response text; logs cost; enforces budget cap."""
    if MOCK:
        return _mock_response(messages, want_json)
    budget.check()
    last_error: BaseException | None = None
    used_model = model
    for candidate in config.model_candidates(model):
        used_model = candidate
        try:
            out, tin, tout = _call(candidate, messages, temperature, max_tokens, want_json)
            break
        except (APIConnectionError, APIStatusError, APITimeoutError) as exc:
            last_error = exc
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("no permitted inference model is configured")
    budget.record(used_model, tin, tout, stage)
    return out


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
def chat_json(model: str, messages: list[Message], stage: str, **kw: Any) -> dict[str, Any]:
    raw = chat(model, messages, stage, want_json=True, **kw)
    # tolerate code fences some models emit despite json mode
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("model returned JSON that is not an object")
    return cast(dict[str, Any], parsed)
