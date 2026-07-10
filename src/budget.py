"""Record model token usage and enforce a configurable development budget guard."""

import json
import os
import time

from . import config

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "budget.jsonl")

# Estimated serverless rates in USD per one million tokens.
# On-demand deployment charges are based on active GPU time and tracked by Fireworks.
RATES: dict[str, dict[str, float]] = {"default": {"in": 0.5, "out": 1.5}}


class BudgetExceeded(RuntimeError):
    pass


def _rate(model: str) -> dict[str, float]:
    return RATES.get(model, RATES["default"])


def spent_usd() -> float:
    if not os.path.exists(LOG_PATH):
        return 0.0
    total = 0.0
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            total += json.loads(line).get("usd", 0.0)
    return total


def check() -> None:
    s = spent_usd()
    if s >= config.BUDGET_USD_CAP:
        raise BudgetExceeded(f"spent ${s:.2f} >= cap ${config.BUDGET_USD_CAP:.2f}")


def record(model: str, tokens_in: int, tokens_out: int, stage: str) -> None:
    r = _rate(model)
    usd = tokens_in / 1e6 * r["in"] + tokens_out / 1e6 * r["out"]
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "stage": stage,
                    "model": model,
                    "in": tokens_in,
                    "out": tokens_out,
                    "usd": round(usd, 6),
                }
            )
            + "\n"
        )


def report() -> str:
    if not os.path.exists(LOG_PATH):
        return "no spend recorded"
    by_stage: dict[str, float] = {}
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_stage[rec["stage"]] = by_stage.get(rec["stage"], 0.0) + rec["usd"]
    lines = [f"  {k:<10} ${v:.4f}" for k, v in sorted(by_stage.items())]
    lines.append(f"  {'TOTAL':<10} ${spent_usd():.4f} / cap ${config.BUDGET_USD_CAP:.2f}")
    return "\n".join(lines)
