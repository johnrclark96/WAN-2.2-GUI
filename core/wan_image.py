from __future__ import annotations

from typing import List

from wan_ps1_engine import (
    attention_context,
    load_pipeline,
    log_vram,
    run_generation,
)


def generate_image_wan(args) -> List[str]:
    """Generate a single-frame image using the WAN 5B pipeline."""
    args.frames = 1
    args.batch_size = 1
    log_vram("before load")
    pipe = load_pipeline(args.model_dir, args.dtype, getattr(args, "no_cache", False))
    log_vram("after load")
    import logging

    logging.info(f"Model dir: {args.model_dir}")
    attn_name, attn_ctx = attention_context(args.attn)
    logging.info(f"Attention backend: {attn_name}")
    try:
        outputs = run_generation(pipe, args, attn_name, attn_ctx)
    finally:
        try:
            import torch

            del pipe
            torch.cuda.empty_cache()
        except Exception:
            pass
    return outputs
