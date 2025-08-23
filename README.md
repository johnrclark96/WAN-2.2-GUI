# WAN 2.2 GUI

Minimal runner and GUI for the WAN 2.2 video generation pipelines.

## Dependency Pins

- `diffusers==0.31.0`
- `transformers==4.49.0` *(pulls `tokenizers` 0.21.x automatically)*

The SDPA/flash-attention selection logic in `wan_ps1_engine.py` is unaffected by these changes.

See [CHANGELOG.md](CHANGELOG.md) for recent updates.
