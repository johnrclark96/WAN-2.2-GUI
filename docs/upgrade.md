# Upgrade Guide

## Checklist

- **Imports** – use `DiffusionPipeline.from_pretrained(..., trust_remote_code=True, torch_dtype=torch.float16)` instead of `WanPipeline` imports.
- **Video VAE** – load via `from_pretrained(..., subfolder='vae', trust_remote_code=True)` and ensure `base_dim`, `z_dim`, `scale_factor_spatial`, and `scale_factor_temporal` exist in `vae/config.json`.
- **Schedulers** – recreate with `SchedulerClass.from_config(pipe.scheduler.config)` and call `scheduler.step()` using keyword arguments.
- **Attention** – set `AttnProcessor2_0()` and use the PyTorch 2.x SDPA API:
  `from torch.nn.attention import sdpa_kernel, SDPBackend` then
  `with sdpa_kernel(SDPBackend.FLASH_ATTENTION): ...`.
- **Dry-run CLI** – `--dry-run`/`--no-model` resolves paths, prints a JSON summary, and exits with status `0`.

## Reverting

If an upgrade fails, reinstall the frozen dependencies:

```bash
pip install -r requirements_frozen.txt
```

The frozen requirements should correspond to the pinned versions described in [env.md](env.md).

