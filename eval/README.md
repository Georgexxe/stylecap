# Evaluation Verification

Current verified baseline:

- 13 deterministic unit and contract tests pass.
- Ruff passes with the configured error, import, upgrade, and bugbear rules.
- Strict mypy passes across all 15 source modules and scripts.
- All three official public clips complete through live Gemma inference.
- The three-clip live suite completes in 92.5 seconds.
- The Docker image completes the mounted `/input` to `/output` workflow in mock mode.
- Image: `linux/amd64`, approximately 606 MB uncompressed.

See `../BENCHMARK.md` for the live model configuration, timings, manual quality review,
cost controls, and stated limitations.
