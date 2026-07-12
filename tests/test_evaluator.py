"""Tests for the official Track 2 evaluator contract."""

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluator import ContractError, EvaluationResult, EvaluationTask, run_evaluation


class EvaluatorTests(unittest.TestCase):
    """Verify exact /input to /output behavior required by the participant guide."""

    def test_task_accepts_official_underscore_style_ids(self) -> None:
        task = EvaluationTask.model_validate(
            {
                "task_id": "v1",
                "video_url": "https://example.com/clip.mp4",
                "styles": [
                    "formal",
                    "sarcastic",
                    "humorous_tech",
                    "humorous_non_tech",
                ],
            }
        )

        self.assertEqual(task.styles[2], "humorous_tech")

    def test_result_rejects_a_missing_requested_style(self) -> None:
        task = EvaluationTask(
            task_id="v1",
            video_url="https://example.com/clip.mp4",
            styles=["formal", "sarcastic"],
        )
        result = EvaluationResult(
            task_id="v1",
            captions={"formal": "A factual caption."},
        )

        with self.assertRaisesRegex(ContractError, "missing requested styles"):
            result.validate_for(task)

    def test_run_evaluation_writes_exact_nested_results_schema(self) -> None:
        payload = [
            {
                "task_id": "v1",
                "video_url": "https://example.com/clip.mp4",
                "styles": ["formal", "sarcastic"],
            }
        ]

        def processor(task: EvaluationTask) -> EvaluationResult:
            return EvaluationResult(
                task_id=task.task_id,
                captions={
                    "formal": "A person walks through a park.",
                    "sarcastic": ("A groundbreaking stroll through a completely ordinary park."),
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "tasks.json"
            output_path = Path(directory) / "results.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            run_evaluation(input_path, output_path, processor=processor)

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [
                    {
                        "task_id": "v1",
                        "captions": {
                            "formal": "A person walks through a park.",
                            "sarcastic": (
                                "A groundbreaking stroll through a completely ordinary park."
                            ),
                        },
                    }
                ],
            )

    def test_processor_failure_still_writes_complete_results_and_continues(self) -> None:
        payload = [
            {
                "task_id": "broken",
                "video_url": "https://example.com/broken.mp4",
                "styles": ["formal", "humorous_tech"],
            },
            {
                "task_id": "working",
                "video_url": "https://example.com/working.mp4",
                "styles": ["formal"],
            },
        ]

        def processor(task: EvaluationTask) -> EvaluationResult:
            if task.task_id == "broken":
                raise RuntimeError("simulated provider rejection")
            return EvaluationResult(
                task_id=task.task_id,
                captions={"formal": "A person waves to the camera."},
            )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "tasks.json"
            output_path = Path(directory) / "results.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            results = run_evaluation(input_path, output_path, processor=processor)
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual([item["task_id"] for item in written], ["broken", "working"])
        self.assertEqual(set(written[0]["captions"]), {"formal", "humorous_tech"})
        self.assertEqual(written[1]["captions"]["formal"], "A person waves to the camera.")
        self.assertEqual([result.task_id for result in results], ["broken", "working"])


if __name__ == "__main__":
    unittest.main()
