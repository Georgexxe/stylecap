# StyleCap Gemma Baseline Benchmark

This document records the original three-call Gemma baseline from July 11. The current
submission evaluator uses a separate 24-frame, single-call serverless scoring path; these
figures are retained as historical product-pipeline evidence and are not current evaluator
performance claims.

## Configuration

- Date: 2026-07-11
- Model: `accounts/fireworks/models/gemma-4-31b-it-nvfp4`
- Serving: Fireworks on-demand, one B200 180 GB GPU, FP4
- Pipeline: frame extraction, one perception call, one batched generation call, one
  batched selection call per clip
- Styles: formal, sarcastic, humorous-tech, humorous-non-tech

## Results

| Workload | Wall time | Outcome |
| --- | ---: | --- |
| One-frame vision probe | 9.4 s | Correctly identified city traffic and autumn trees |
| One official clip | 19.3 s | Valid Track 2 JSON with all four styles |
| All three official clips | 92.5 s | Three complete results; two clips required fresh download and perception |

The three-clip average was 30.8 seconds per clip. A linear 12-clip projection is about
6.2 minutes, below the 10-minute evaluator limit, but network variance and hidden-video
complexity remain risks. Each individual model request observed during this run completed
within the documented 30-second request limit.

## Manual Quality Review

- Urban clip: correctly captured time-lapse traffic, yellow autumn trees, buses, and
  high-rise buildings.
- Kitten clip: correctly captured an orange kitten emerging from foliage and approaching
  the camera.
- Office clip: correctly captured a woman alternating between typing and looking at a
  monitor with a puzzled expression.
- Each clip showed clear separation between factual, sarcastic, technical-humor, and
  everyday-humor captions.

The first run repeated uncertain incidental signage. Prompts now require clearly legible
text and reject signage that is not central to the visible action. This correction has unit,
lint, and strict type checks, but requires another live run before claiming model-level
validation.

## Cost Control

The original deployment was billed at $10 per active GPU-hour. The current evaluator no
longer references account-specific on-demand infrastructure and uses serverless endpoints.

## Limitations

- These are public development clips, not the hidden evaluation set.
- Quality observations are manual, not official LLM-judge scores.
- The one-clip run reused downloaded media and extracted frames.
- A scaled-to-zero deployment must be warmed before latency-sensitive evaluation.
