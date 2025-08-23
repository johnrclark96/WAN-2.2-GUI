# Changelog

## Unreleased

- Fix AutoModel import by using `transformers.AutoModel` and `AutoTokenizer` for text encoders.
- Load VAE with `diffusers.AutoencoderKL`.
- Added defensive import check for `transformers`.
- Pin dependencies: `diffusers==0.31.0`, `transformers==4.49.0`.
- Documented that SDPA/flash-attn selection remains unchanged.
