# Changelog

## Unreleased

- Add `--dry-run`/`--no-model` option, attention backend selection, and VAE config validation.
- Report `[RESULT] OK` or `[RESULT] FAIL` with traceback for pipeline init.
- Add CI workflow running `compileall`, `ruff`, `mypy`, and CLI sanity checks.
- Fix AutoModel import by using `transformers.AutoModel` and `AutoTokenizer` for text encoders.
- Load VAE with `diffusers.AutoencoderKL`.
- Added defensive import check for `transformers`.
- Pin dependencies: `diffusers==0.35.*`, `transformers==4.44.*`,
  `accelerate==0.34.*` (Torch cu121 2.4.x).
- Documented that SDPA/flash-attn selection remains unchanged.
- Cleaned up imports and exception handling in the UI and engine runner.
- Fix cache path handling in `wan_runner.ps1` and ensure cache directories exist.
- Remove unused variable from `tools/install_wan_deps.ps1` and gate flash-attn by compute capability.
- Introduce basic engine toggle with Official command wiring and path persistence.
- Add smoketest coverage and log parser for Official engine output.
