"""Behavioral tests for submission output validation."""

import tempfile
import unittest
from pathlib import Path

from src.emit import FormatError, write
from src.schemas import FinalCaption


def captions_for(clip_id: str) -> list[FinalCaption]:
    """Build one complete four-style result for a clip."""
    return [
        FinalCaption(clip_id=clip_id, style=style, caption=f"{style} caption")
        for style in (
            "formal",
            "sarcastic",
            "humorous_tech",
            "humorous_non_tech",
        )
    ]


class EmitTests(unittest.TestCase):
    """Ensure emitted files cannot silently omit evaluation records."""

    def test_write_rejects_an_entirely_missing_clip(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaisesRegex(FormatError, "missing clips"):
                write(
                    captions_for("clip-a"),
                    out_dir,
                    expected_clip_ids={"clip-a", "clip-b"},
                )

    def test_write_rejects_duplicate_styles(self) -> None:
        records = captions_for("clip-a")
        records.append(records[0].model_copy())

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaisesRegex(FormatError, "duplicate style"):
                write(records, out_dir, expected_clip_ids={"clip-a"})

    def test_write_emits_a_complete_jsonl_file(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            path = Path(
                write(
                    captions_for("clip-a"),
                    out_dir,
                    expected_clip_ids={"clip-a"},
                )
            )

            self.assertTrue(path.is_file())
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
