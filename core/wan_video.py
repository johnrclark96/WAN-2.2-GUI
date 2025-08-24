"""Video generation helpers for WAN 2.2 5B pipeline."""

from __future__ import annotations

from typing import Any, List

import wan_ps1_engine as engine


def generate_video_wan(args: Any, pipe: Any | None = None) -> List[str]:
    """Generate video (or single PNG) using the 5B pipeline.

    Parameters
    ----------
    args:
        Argparse-style namespace with generation parameters.
    pipe:
        Optional preloaded pipeline for testing.
    """

    engine.log_vram("before load")
    if pipe is None:
        pipe = engine.load_pipeline(args.model_dir, args.dtype)
    engine.log_vram("after load")
    attn_name, attn_ctx = engine.attention_context(args.attn)
    print(f"[INFO] Attention backend: {attn_name}")
    try:
        outputs = engine.run_generation(pipe, args, attn_name, attn_ctx)
        engine.log_vram("after run")
        return outputs
    finally:
        if engine.torch is not None:
            try:
                engine.torch.cuda.empty_cache()
            except Exception:
                pass

