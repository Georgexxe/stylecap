# StyleCap Submission

## Title

StyleCap - Grounded Four-Style Video Captioning with Gemma

## Project links

- Repository: https://github.com/Georgexxe/stylecap
- Hosted demo: https://stylecap-gemma.streamlit.app/
- Public container: `ghcr.io/georgexxe/stylecap:latest`

## Short description

StyleCap turns unseen videos into accurate captions in four unmistakable voices: formal,
sarcastic, humorous-tech, and humorous-non-tech. A compact Gemma-centered pipeline first
extracts grounded scene facts, then generates four candidates per voice and selects all
four styles in only three model calls per clip.

## Long description

Humorous video captioning creates a structural conflict: increasing creativity often
increases hallucination. StyleCap separates observation from writing. A vision-capable
model converts sampled video frames into a factual scene sheet containing entities,
actions, timeline beats, visible text, and grounded humor hooks. Gemma then produces four
candidates for every requested voice in one batched call. A final judge call selects the
candidate that best balances factual accuracy and tone fidelity.

The evaluation path is designed for the published hidden benchmark rather than the three
example clips. It accepts arbitrary evaluator video URLs, enforces the exact Track 2 JSON
contract, supports all four underscore style IDs, validates output completeness, and runs
inside a public Linux AMD64 container. Local development includes deterministic mock
inference, strict typing, contract tests, bounded downloads, cached frame extraction, and a
Streamlit demo that uses the production pipeline.

## Technology tags

Gemma, Fireworks AI, video captioning, multimodal AI, Streamlit, Docker

## Live-fire evidence

- Model: Gemma 4 31B IT NVFP4 on a dedicated Fireworks FP4 deployment.
- Public coverage: all three official clips completed with all four required styles.
- One-clip evaluator run: 19.3 seconds with cached media and fresh model calls.
- Three-clip suite: 92.5 seconds, including two fresh downloads and perception passes.
- Manual review found correct central actions and distinct styles on all three clips.
- The deployment costs $10 per active GPU-hour and was verified at zero replicas after testing.

See `BENCHMARK.md` for method, limitations, and representative observations.

## Judge access

The hosted demo is public and leads with live inference from an uploaded clip or direct
public video URL. It keeps the source video beside the workflow, exposes grounding evidence,
and returns four downloadable caption styles. A separate benchmark view provides immediate
recorded evidence. The public container implements the published evaluator contract and
accepts standard Fireworks configuration through environment variables.
