"""Run the Track 2 evaluator against an explicit on-demand deployment."""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment")
    parser.add_argument("--input", type=Path, default=Path("examples/tasks.json"))
    parser.add_argument("--output", type=Path, default=Path("live-results.json"))
    args = parser.parse_args()

    os.environ["STYLECAP_GEMMA_DEPLOYMENT"] = args.deployment
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src import config, evaluator, pipeline

    config.validate_runtime()
    results = evaluator.run_evaluation(args.input, args.output, processor=pipeline.process_task)
    print(f"Wrote {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
