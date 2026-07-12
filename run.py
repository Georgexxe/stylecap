"""StyleCap CLI.

  process --clips DIR --out DIR   full pipeline over real clips
  demo                            end-to-end dry run with mock LLM (no clips, no credits)
  report                          budget spend summary

Set STYLECAP_MOCK=1 to run any command without API calls.
"""

import glob
import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, os.path.dirname(__file__))
from src import budget, compact, config, emit, evaluator, llm, pipeline  # noqa: E402
from src.schemas import FactSheet, HumorHook, TimelineBeat  # noqa: E402

app = typer.Typer(no_args_is_help=True)


@app.command()
def process(clips: str = typer.Option(...), out: str = typer.Option("out")) -> None:
    """Run the full pipeline over a directory of video clips."""
    paths = sorted(
        p
        for ext in ("mp4", "mov", "mkv", "webm")
        for p in glob.glob(os.path.join(clips, f"*.{ext}"))
    )
    if not paths:
        typer.echo(f"no clips found in {clips}")
        raise typer.Exit(1)
    all_final = []
    expected_clip_ids = {os.path.splitext(os.path.basename(path))[0] for path in paths}
    for p in paths:
        typer.echo(f"clip: {os.path.basename(p)}")
        result = pipeline.process_clip(p)
        final = result.captions
        all_final.extend(final)
        for fc in final:
            typer.echo(f"  [{fc.style}] {fc.caption}")
    path = emit.write(all_final, out, expected_clip_ids=expected_clip_ids)
    typer.echo(f"\nwrote {path}\n\nbudget:\n{budget.report()}")


@app.command()
def evaluate(
    input_path: str = typer.Option("/input/tasks.json", "--input"),
    output_path: str = typer.Option("/output/results.json", "--output"),
    mock: bool = typer.Option(False, "--mock"),
) -> None:
    """Run the official Track 2 evaluator file contract."""
    if mock:
        llm.MOCK = True
    processor = pipeline.process_task
    if not llm.MOCK:
        try:
            config.validate_runtime()
        except RuntimeError as exc:
            typer.echo(f"runtime configuration unavailable: {exc}; writing fallbacks", err=True)

            def unavailable_processor(
                task: evaluator.EvaluationTask,
                cause: RuntimeError = exc,
            ) -> evaluator.EvaluationResult:
                raise cause

            processor = unavailable_processor
    results = evaluator.run_evaluation(
        Path(input_path),
        Path(output_path),
        processor=processor,
        max_workers=config.EVALUATION_WORKERS,
    )
    typer.echo(f"wrote {len(results)} task result(s) to {output_path}")


@app.command()
def demo() -> None:
    """Verify pipeline wiring with a synthetic clip description and mock inference."""
    os.environ["STYLECAP_MOCK"] = "1"
    llm.MOCK = True
    facts = FactSheet(
        clip_id="demo-cat",
        setting="a kitchen counter",
        entities=["a cat", "three glasses"],
        actions=["cat walks along counter", "cat knocks third glass off while facing camera"],
        timeline=[
            TimelineBeat(t="0:05", event="cat steps onto counter"),
            TimelineBeat(t="0:32", event="third glass falls"),
        ],
        on_screen_text=[],
        audio_events=["off-screen voice: 'Whiskers, no'"],
        mood="chaotic-domestic",
        humor_hooks=[
            HumorHook(
                observation="cat stares at camera while pushing glass",
                why_funny="premeditation plus eye contact",
            )
        ],
    )
    final = compact.run(facts)
    typer.echo(f"selected {len(final)} captions")
    for fc in final:
        typer.echo(f"  [{fc.style}] {fc.caption}")
    path = emit.write(final, "out-demo", expected_clip_ids={facts.clip_id})
    typer.echo(f"wrote {path}\npipeline wiring OK")


@app.command()
def report() -> None:
    """Budget spend summary."""
    typer.echo(budget.report())


if __name__ == "__main__":
    app()
