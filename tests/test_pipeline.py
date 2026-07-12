"""Behavioral tests for the reusable single-clip pipeline."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src import pipeline
from src.evaluator import EvaluationTask
from src.schemas import FactSheet, FinalCaption


class PipelineTests(unittest.TestCase):
    """Ensure the CLI and Streamlit app can share one execution path."""

    def test_process_clip_runs_each_stage_once(self) -> None:
        facts = FactSheet(
            clip_id="clip-a",
            setting="a room",
            entities=["a person"],
            actions=["the person waves"],
            timeline=[{"t": "0:01", "event": "the person waves"}],
            mood="neutral",
        )
        final = [FinalCaption(clip_id="clip-a", style="formal", caption="A person waves.")]

        with (
            patch.object(
                pipeline.ingest,
                "run",
                return_value={"clip_id": "clip-a", "frames": ["frame-a.jpg"]},
            ) as ingest_run,
            patch.object(pipeline.perceive, "run", return_value=facts) as perceive_run,
            patch.object(pipeline.compact, "run", return_value=final) as compact_run,
        ):
            result = pipeline.process_clip("clip-a.mp4")

        self.assertEqual(result.facts, facts)
        self.assertEqual(result.captions, final)
        ingest_run.assert_called_once_with("clip-a.mp4")
        perceive_run.assert_called_once()
        compact_run.assert_called_once_with(facts, None, frame_paths=["frame-a.jpg"])

    def test_process_task_downloads_video_and_returns_nested_captions(self) -> None:
        task = EvaluationTask(
            task_id="v1",
            video_url="https://example.com/clip.mp4",
            styles=["formal", "sarcastic"],
        )
        captions = [
            FinalCaption(clip_id="evaluation", style="formal", caption="A person waves."),
            FinalCaption(
                clip_id="evaluation",
                style="sarcastic",
                caption="A historic achievement in hand movement.",
            ),
        ]

        with (
            patch.object(
                pipeline.ingest,
                "download_video",
                return_value=Path("cache/downloads/clip.mp4"),
            ) as download,
            patch.object(
                pipeline.ingest,
                "run",
                return_value={"clip_id": "downloaded", "frames": ["frame-a.jpg"]},
            ) as ingest_run,
            patch.object(pipeline.scoring, "run", return_value=captions) as score,
        ):
            result = pipeline.process_task(task)

        self.assertEqual(result.task_id, "v1")
        self.assertEqual(result.captions["formal"], "A person waves.")
        download.assert_called_once()
        ingest_run.assert_called_once_with(os.path.normpath("cache/downloads/clip.mp4"))
        score.assert_called_once_with(["frame-a.jpg"], ["formal", "sarcastic"])


if __name__ == "__main__":
    unittest.main()
