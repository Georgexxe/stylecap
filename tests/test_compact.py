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

    def test_scoring_prompt_excludes_inferred_humor_metadata(self) -> None:
        sheet = facts().model_copy(
            update={
                "mood": "secretly frustrated",
                "humor_hooks": [
                    {
                        "observation": "the runner pauses",
                        "why_funny": "the runner is late for a meeting",
                    }
                ],
            }
        )
        generation = {"candidates": {"formal": ["zero", "one", "two"]}}
        selection = {"selected": {"formal": 0}}

        with patch.object(
            compact.llm, "chat_json", side_effect=[generation, selection]
        ) as chat_json:
            compact.run(sheet, ["formal"])

        generation_prompt = chat_json.call_args_list[0].args[1][1]["content"]
        self.assertNotIn("secretly frustrated", generation_prompt)
        self.assertNotIn("late for a meeting", generation_prompt)
        self.assertIn("the runner ties a shoe", generation_prompt)

    def test_selection_receives_representative_video_frames(self) -> None:
        generation = {"candidates": {"formal": ["zero", "one", "two"]}}
        selection = {"selected": {"formal": 0}}

        with (
            patch.object(compact.llm, "chat_json", side_effect=[generation, selection])
            as chat_json,
            patch.object(
                compact.ingest,
                "frame_to_data_url",
                side_effect=lambda path: f"data:image/jpeg;base64,{path}",
            ),
        ):
            compact.run(
                facts(),
                ["formal"],
                frame_paths=[f"frame-{index}.jpg" for index in range(8)],
            )

        selection_content = chat_json.call_args_list[1].args[1][1]["content"]
        images = [item for item in selection_content if item["type"] == "image_url"]
        self.assertEqual(len(images), compact.MAX_SELECTION_FRAMES)
        self.assertIn("frame-0.jpg", images[0]["image_url"]["url"])
        self.assertIn("frame-7.jpg", images[-1]["image_url"]["url"])


if __name__ == "__main__":
    unittest.main()
