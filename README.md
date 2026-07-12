# StyleCap

![StyleCap cover](assets/cover-v2.png)

StyleCap generates grounded captions for video clips in four requested voices:
`formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech`.

**Live demo:** https://stylecap-gemma.streamlit.app/

The scoring container uses a benchmark-focused path:

1. Sample 24 frames uniformly across the complete clip.
2. Send the full temporal sequence to serverless MiniMax M3 in one multimodal call.
3. Validate caption length, completeness, and style distinctness before writing results.
4. Process up to four evaluator clips concurrently while preserving task order.

Captions contain 2-3 complete sentences so the LLM judge receives both broad factual
coverage and an unmistakable tone signal. The public product demo retains the separate
Gemma fact-sheet and caption-selection workflow for transparent grounding.
The Docker image installs only `requirements-runtime.txt`; Streamlit and its analytics
dependencies remain outside the scoring image.

## Evaluator contract

The container reads `/input/tasks.json` on startup and writes `/output/results.json`
before exiting. Style IDs use underscores exactly as published in the participant guide.
During judging it reads `ALLOWED_MODELS` and uses only model IDs authorized by the
evaluator. Results are written atomically before remote inference begins and updated after
each successful task, so a provider or clip failure cannot crash the whole batch or leave
the evaluator without valid JSON.

Input:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Output:

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Configuration

Never commit credentials. Copy `.env.example` for local development and set:

- `FIREWORKS_API_KEY`: your Fireworks API key.
- `ALLOWED_MODELS`: evaluator-supplied JSON array or comma-separated model allow-list.
  When present, it takes precedence over every local model override.
- `STYLECAP_SCORING_MODEL`: evaluator model; defaults to serverless MiniMax M3.
- `STYLECAP_GEMMA_DEPLOYMENT`: optional Gemma model override for the public demo pipeline.
- `STYLECAP_SERVERLESS_FALLBACK_MODEL`: evaluator fallback; defaults to Gemma 4 31B IT.
- `STYLECAP_EVALUATION_WORKERS`: concurrent evaluator clips; defaults to four.
- `STYLECAP_PERCEPTION_MODEL`, `STYLECAP_STYLE_MODEL`, and
  `STYLECAP_JUDGE_MODELS`: optional experiment-only overrides.
- `STYLECAP_ENABLE_ASR=1`: optional local Whisper transcription. It is disabled by default to avoid cold-start downloads during evaluation.

The submission uses serverless endpoints only. It does not require or reference an
account-specific dedicated deployment.

## Local verification

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run.py demo
```

Run the official file contract in mock mode:

```powershell
.\.venv\Scripts\python.exe run.py evaluate --mock --input examples/tasks.json --output out/results.json
```

Run the Streamlit demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The public Streamlit deployment accepts an uploaded clip or a direct public video URL,
keeps the source video visible during review, and returns grounding evidence with four
downloadable captions. It includes the system `ffmpeg` package required for video decoding.
Live inference uses serverless Fireworks models and requires an active API account.

## Docker

Build the required Linux AMD64 image:

```bash
docker buildx build --platform linux/amd64 -t ghcr.io/georgexxe/stylecap:latest --push .
```

Pushes to `main` run the contract tests, lint checks, and a Linux AMD64 image build through
GitHub Actions. Publishing remains an explicit release step because the existing GHCR
package does not grant repository workflow tokens write access.

Local evaluator run:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  ghcr.io/georgexxe/stylecap:latest
```

## Repository layout

- `src/evaluator.py`: published Track 2 input/output contract.
- `src/ingest.py`: bounded URL download, frame extraction, and optional ASR.
- `src/perceive.py`: grounded fact-sheet extraction.
- `src/compact.py`: two-call batched generation and frame-verified selection.
- `app.py`: Streamlit demo using the production pipeline.
- `tests/`: contract, download, selection, and pipeline tests.

## License

MIT
