"""Tests for the low-latency Track 2 caption path."""

import unittest
from unittest.mock import patch

from src import compact
from src.schemas import FactSheet


def facts() -> FactSheet:
    """Return a minimal fact sheet for compact generation tests."""
    return FactSheet(
        clip_id="clip-a",
        setting="a city park",
        entities=["a runner"],
        actions=["the runner ties a shoe"],
        timeline=[{"t": "0:03", "event": "the runner ties a shoe"}],
        mood="calm",
    )


class CompactTests(unittest.TestCase):
    """Ensure all styles are generated and selected in only two model calls."""

    def test_run_batches_generation_and_selection(self) -> None:
        generation = {
            "candidates": {
                "formal": ["formal zero", "formal one", "formal two"],
                "sarcastic": ["sarcastic zero", "sarcastic one", "sarcastic two"],
            }
        }
        selection = {"selected": {"formal": 1, "sarcastic": 2}}

        with patch.object(
            compact.llm, "chat_json", side_effect=[generation, selection]
        ) as chat_json:
            results = compact.run(facts(), ["formal", "sarcastic"])

        self.assertEqual(chat_json.call_count, 2)
        self.assertEqual(
            {result.style: result.caption for result in results},
            {"formal": "formal one", "sarcastic": "sarcastic two"},
        )

    def test_run_rejects_an_out_of_range_selection(self) -> None:
        generation = {"candidates": {"formal": ["zero", "one", "two"]}}
        selection = {"selected": {"formal": 9}}

        with patch.object(compact.llm, "chat_json", side_effect=[generation, selection]):
            with self.assertRaisesRegex(compact.CompactCaptionError, "formal"):
                compact.run(facts(), ["formal"])


if __name__ == "__main__":
    unittest.main()
