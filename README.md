# WAN 2.2 GUI

Minimal runner and GUI for the WAN 2.2 video generation pipelines.

## Quick start (dry-run)

Run the CLI without downloading models to verify the environment:

```bash
python wan_ps1_engine.py --dry-run --attn auto --mode t2v --frames 8 --width 512 --height 288
```

## Runtime setup

See [docs/env.md](docs/env.md) for pinned dependency versions and installation notes.

See [CHANGELOG.md](CHANGELOG.md) for recent updates.
