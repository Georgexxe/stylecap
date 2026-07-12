"""CLI regression tests for evaluator startup failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

import run


class EvaluateCliTests(unittest.TestCase):
    def test_missing_api_key_exits_zero_with_contract_safe_output(self) -> None:
        payload = [
            {
                "task_id": "v1",
                "video_url": "https://example.com/clip.mp4",
                "styles": ["formal", "sarcastic"],
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "tasks.json"
            output_path = Path(directory) / "results.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            with (
                patch.object(run.llm, "MOCK", False),
                patch.object(
                    run.config,
                    "validate_runtime",
                    side_effect=RuntimeError("missing FIREWORKS_API_KEY"),
                ),
            ):
                result = CliRunner().invoke(
                    run.app,
                    [
                        "evaluate",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ],
                )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(written[0]["task_id"], "v1")
        self.assertEqual(set(written[0]["captions"]), {"formal", "sarcastic"})


if __name__ == "__main__":
    unittest.main()
