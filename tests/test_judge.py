"""Behavioral tests for candidate selection."""

import unittest
from unittest.mock import patch

from src import config, judge
from src.schemas import Candidate, FactSheet


def facts() -> FactSheet:
    """Return a minimal grounded fact sheet."""
    return FactSheet(
        clip_id="clip-a",
        setting="a room",
        entities=["a person"],
        actions=["the person waves"],
        timeline=[{"t": "0:01", "event": "the person waves"}],
        mood="neutral",
    )


class JudgeTests(unittest.TestCase):
    """Ensure judging either returns complete output or reports a hard failure."""

    def test_run_raises_when_every_candidate_fails_accuracy(self) -> None:
        candidates = [
            Candidate(clip_id="clip-a", style=style, caption="invented event")
            for style in config.STYLES
        ]

        def reject(_facts: FactSheet, candidate: Candidate) -> Candidate:
            candidate.accuracy_pass = False
            candidate.unverified_claims = ["invented event"]
            return candidate

        with (
            patch.object(judge, "accuracy_gate", side_effect=reject),
            patch.object(judge.stylize, "run", return_value=candidates),
        ):
            with self.assertRaisesRegex(judge.CaptionSelectionError, "clip-a"):
                judge.run(facts(), candidates)


if __name__ == "__main__":
    unittest.main()
