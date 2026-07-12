# StyleCap Submission

## Title

StyleCap - Grounded Four-Style Video Captioning with Gemma

## Project links

- Repository: https://github.com/Georgexxe/stylecap
- Hosted demo: https://stylecap-gemma.streamlit.app/
- Public container: `ghcr.io/georgexxe/stylecap:latest`

## Short description

StyleCap turns unseen videos into accurate captions in four unmistakable voices: formal,
sarcastic, humorous-tech, and humorous-non-tech. Its scoring container samples 24 frames
across each clip and generates all requested styles in one multimodal Fireworks call.

## Long description

Humorous video captioning creates a structural conflict: increasing creativity often
increases hallucination. StyleCap addresses this with two purpose-built execution paths.
The public Gemma experience exposes a factual scene sheet and grounded caption selection.
The automated evaluator sends 24 uniformly sampled frames directly to MiniMax M3, requests
2-3 sentence captions with strong factual coverage, validates every style, and processes
multiple clips concurrently within the benchmark time limit.

The evaluation path is designed for the published hidden benchmark rather than the three
example clips. It accepts arbitrary evaluator video URLs, enforces the exact Track 2 JSON
contract, supports all four underscore style IDs, validates output completeness, and runs
inside a public Linux AMD64 container. Local development includes deterministic mock
inference, strict typing, contract tests, bounded downloads, cached frame extraction, and a
Streamlit demo that uses the production pipeline.

## Technology tags

Gemma, MiniMax M3, Fireworks AI, video captioning, multimodal AI, Streamlit, Docker

## Verification evidence

- The exact three-task evaluator contract passes in the Linux submission container.
- Every official public clip supplies 24 uniformly distributed frames.
- Four concurrent workers preserve original task order and exact requested style keys.
- Caption validation rejects malformed, short, empty, and near-duplicate outputs.
- Unit, contract, lint, and strict type checks pass.

`BENCHMARK.md` records the earlier Gemma baseline and its limitations.

## Judge access

The hosted demo is public and leads with live inference from an uploaded clip or direct
public video URL. It keeps the source video beside the workflow, exposes grounding evidence,
and returns four downloadable caption styles. A separate benchmark view provides immediate
recorded evidence. The public container implements the published evaluator contract and
accepts standard Fireworks configuration through environment variables.
