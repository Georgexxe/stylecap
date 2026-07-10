"""Runtime configuration for local development and the evaluation container."""

import os

GEMMA_DEPLOYMENT = os.environ.get("STYLECAP_GEMMA_DEPLOYMENT", "")
PERCEPTION_MODEL = os.environ.get("STYLECAP_PERCEPTION_MODEL", GEMMA_DEPLOYMENT)
STYLE_MODEL = os.environ.get("STYLECAP_STYLE_MODEL", GEMMA_DEPLOYMENT)
JUDGE_MODELS = [
    model.strip()
    for model in os.environ.get("STYLECAP_JUDGE_MODELS", GEMMA_DEPLOYMENT).split(",")
    if model.strip()
]
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
TARGET_FPS = 0.75
MAX_FRAMES = 12
SCENE_CHANGE_THRESHOLD = 0.3
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
    if not GEMMA_DEPLOYMENT:
        missing.append("STYLECAP_GEMMA_DEPLOYMENT")
    if missing:
        raise RuntimeError(f"missing runtime configuration: {', '.join(missing)}")
