"""Runtime configuration for local development and the evaluation container."""

import json
import os
from typing import Any


def parse_allowed_models(raw: str | None = None) -> list[str]:
    """Parse the evaluator's model allow-list in JSON or comma-separated form."""
    value = os.environ.get("ALLOWED_MODELS", "") if raw is None else raw
    value = value.strip()
    if not value:
        return []

    try:
        decoded: Any = json.loads(value)
    except json.JSONDecodeError:
        decoded = None

    if isinstance(decoded, list):
        return list(
            dict.fromkeys(
                item.strip()
                for item in decoded
                if isinstance(item, str) and item.strip()
            )
        )
    if isinstance(decoded, str) and decoded.strip():
        return [decoded.strip()]
    return list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


ALLOWED_MODELS = parse_allowed_models()

DEFAULT_GEMMA_MODEL = "accounts/fireworks/models/gemma-4-31b-it"
GEMMA_DEPLOYMENT = os.environ.get("STYLECAP_GEMMA_DEPLOYMENT", DEFAULT_GEMMA_MODEL)
SERVERLESS_FALLBACK_MODEL = os.environ.get(
    "STYLECAP_SERVERLESS_FALLBACK_MODEL",
    DEFAULT_GEMMA_MODEL,
)
SCORING_MODEL = os.environ.get(
    "STYLECAP_SCORING_MODEL",
    "accounts/fireworks/models/minimax-m3",
)


def _permitted_model(requested: str) -> str:
    """Use local configuration only when the evaluator has not supplied a policy."""
    if not ALLOWED_MODELS:
        return requested
    return requested if requested in ALLOWED_MODELS else ALLOWED_MODELS[0]


PERCEPTION_MODEL = _permitted_model(
    os.environ.get("STYLECAP_PERCEPTION_MODEL", GEMMA_DEPLOYMENT)
)
STYLE_MODEL = _permitted_model(os.environ.get("STYLECAP_STYLE_MODEL", GEMMA_DEPLOYMENT))
_requested_judges = [
    model.strip()
    for model in os.environ.get("STYLECAP_JUDGE_MODELS", GEMMA_DEPLOYMENT).split(",")
    if model.strip()
]
JUDGE_MODELS = (
    [model for model in _requested_judges if model in ALLOWED_MODELS]
    or ALLOWED_MODELS[:1]
    if ALLOWED_MODELS
    else _requested_judges
)
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Generation
K_CANDIDATES = 2
TEMPS = {
    "formal": 0.4,
    "sarcastic": 0.8,
    "humorous_tech": 0.8,
    "humorous_non_tech": 0.8,
}
MAX_REGEN_ROUNDS = 2

# Ingest
TARGET_FPS = 1.25
MAX_FRAMES = 24
EVALUATION_WORKERS = int(os.environ.get("STYLECAP_EVALUATION_WORKERS", "4"))
SCENE_CHANGE_THRESHOLD = 0.3
INGEST_CACHE_VERSION = "v4"
ENABLE_ASR = os.environ.get("STYLECAP_ENABLE_ASR", "0") == "1"
WHISPER_MODEL = os.environ.get("STYLECAP_WHISPER_MODEL", "small")

# Budget guard
BUDGET_USD_CAP = 40.0  # hard stop at 80% of $50
CACHE_DIR = "cache"


def validate_runtime() -> None:
    """Fail before processing when required real-inference settings are missing."""
    missing = []
    if not os.environ.get("FIREWORKS_API_KEY"):
        missing.append("FIREWORKS_API_KEY")
    if missing:
        raise RuntimeError(f"missing runtime configuration: {', '.join(missing)}")


def model_candidates(primary: str) -> list[str]:
    """Return unique, policy-compliant models to try for an inference call."""
    if ALLOWED_MODELS:
        candidates = (
            [primary, *ALLOWED_MODELS]
            if primary in ALLOWED_MODELS
            else ALLOWED_MODELS
        )
    else:
        candidates = [primary, SERVERLESS_FALLBACK_MODEL]
    return list(dict.fromkeys(model for model in candidates if model))
