"""Tests for model-aware perception caching."""

import unittest
from unittest.mock import patch

from src import perceive


class PerceptionCacheTests(unittest.TestCase):
    def test_cache_path_changes_by_model_and_inference_mode(self) -> None:
        media = {
            "clip_id": "clip-a",
            "clip_hash": "abc123",
            "frames": [],
            "transcript": [],
        }

        with (
            patch.object(perceive.config, "PERCEPTION_MODEL", "model-a"),
            patch.object(perceive.llm, "MOCK", False),
        ):
            model_a = perceive._cache_path(media)
        with (
            patch.object(perceive.config, "PERCEPTION_MODEL", "model-b"),
            patch.object(perceive.llm, "MOCK", False),
        ):
            model_b = perceive._cache_path(media)
        with (
            patch.object(perceive.config, "PERCEPTION_MODEL", "model-a"),
            patch.object(perceive.llm, "MOCK", True),
        ):
            mock_model_a = perceive._cache_path(media)

        self.assertNotEqual(model_a, model_b)
        self.assertNotEqual(model_a, mock_model_a)
        self.assertTrue(model_a.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
