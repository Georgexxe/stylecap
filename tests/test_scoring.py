"""Tests for the leaderboard-tuned direct caption generation path."""

import unittest
from unittest.mock import patch

from src import scoring


class ScoringTests(unittest.TestCase):
    def test_validate_rejects_short_captions(self) -> None:
        with self.assertRaisesRegex(scoring.ScoringCaptionError, "expected 24-70 words"):
            scoring._validate({"formal": "A person waves."}, ["formal"])

    def test_run_sends_all_frames_and_preserves_requested_styles(self) -> None:
        payload = {
            "formal": (
                "A person in a blue jacket walks through a city plaza while carrying a red "
                "bag. Several pedestrians pass in the opposite direction as the person crosses "
                "the open paved area and approaches a row of trees."
            ),
            "sarcastic": (
                "A traveler carries a bright red bag across the plaza with the solemn focus of "
                "someone completing history's most demanding sidewalk mission. Nearby pedestrians "
                "somehow continue their day without pausing to applaud this remarkable crossing."
            ),
        }
        with (
            patch.object(scoring.ingest, "frame_to_data_url", side_effect=["a", "b", "c"]),
            patch.object(scoring.llm, "chat_json", return_value=payload) as chat,
        ):
            result = scoring.run(["1.jpg", "2.jpg", "3.jpg"], ["formal", "sarcastic"])

        self.assertEqual([caption.style for caption in result], ["formal", "sarcastic"])
        messages = chat.call_args.args[1]
        self.assertEqual(len(messages[1]["content"]), 4)


if __name__ == "__main__":
    unittest.main()
