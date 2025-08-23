# Changelog

## Unreleased

- Add `--dry-run`/`--no-model` option, attention backend selection, and VAE config validation.
- Report `[RESULT] OK` or `[RESULT] FAIL` with traceback for pipeline init.
- Add CI workflow running `py_compile`, `ruff`, `mypy`, and CLI sanity checks.
- Fix AutoModel import by using `transformers.AutoModel` and `AutoTokenizer` for text encoders.
- Load VAE with `diffusers.AutoencoderKL`.
- Added defensive import check for `transformers`.
- Pin dependencies: `diffusers==0.31.0`, `transformers==4.49.0`.
- Documented that SDPA/flash-attn selection remains unchanged.
- Cleaned up imports and exception handling in the UI and engine runner.
