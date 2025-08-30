"""Image generation helper for WAN 2.2 5B pipeline."""

from __future__ import annotations

from typing import Any, List, Tuple

import wan_ps1_engine as engine


def generate_image_wan(args: Any, pipe: Any | None = None) -> Tuple[List[str], str]:
    """Generate a single PNG using the 5B pipeline."""

    args.frames = 1
    engine.log_vram("before load")
    if pipe is None:
        pipe = engine.load_pipeline(args.model_dir, args.dtype, args.offload)
    engine.log_vram("after load")
    attn_name, attn_ctx = engine.attention_context(args.attn, pipe)
    print(f"[INFO] Attention backend: {attn_name}")
    try:
        outputs = engine.run_generation(pipe, args, attn_name, attn_ctx)
        engine.log_vram("after run")
        return outputs, attn_name
    finally:
        if engine.torch is not None:
            try:
                engine.torch.cuda.empty_cache()
            except Exception:
                pass

