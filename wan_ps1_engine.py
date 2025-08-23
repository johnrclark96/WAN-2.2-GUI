#!/usr/bin/env python
"""CLI engine for WAN 2.2.

This script sanitizes arguments and calls the Diffusers ``WanPipeline``.
It focuses on robust input validation and predictable GPU/offload
behaviour.  Errors are reported via JSON sidecars and a concise
``[RESULT]`` line on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

try:
    import torch
except Exception:  # pragma: no cover - torch may be absent in tests
    torch = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - pillow may be absent in tests
    Image = None  # type: ignore

# Diffusers imports are intentionally placed inside ``load_pipeline`` so a dry
# run can execute without the package installed.


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def snap32(x: int) -> int:
    """Snap ``x`` down to the nearest multiple of 32."""
    return max(32, int(x) // 32 * 32)


def log_vram(label: str) -> None:
    """Print VRAM usage for debugging."""
    if torch is None or not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()  # type: ignore[arg-type]
    used = (total - free) / 1024**2
    print(f"[INFO] {label} VRAM: {used:.0f} MiB used / {total/1024**2:.0f} MiB total")


def attention_context(pref: str):
    """Return (name, context manager) for the requested attention backend."""
    if torch is None:
        return "math", nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        return "math", nullcontext()

    if pref in {"auto", "flash"}:
        try:
            return "flash", sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        except Exception:
            if pref == "flash":
                raise
    if pref in {"auto", "sdpa"}:
        try:
            return "sdpa", sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        except Exception:
            pass
    try:
        return "math", sdpa_kernel(SDPBackend.MATH)
    except Exception:
        return "math", nullcontext()


def load_image(path: str, width: int, height: int) -> Image.Image:
    if Image is None:
        raise RuntimeError("pillow is required for image-based modes")
    img = Image.open(path).convert("RGB")
    return img.resize((width, height), Image.BICUBIC)


def save_sidecar(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# pipeline loader and runner
# ---------------------------------------------------------------------------

def load_pipeline(model_dir: str, dtype: str):
    if torch is None:
        raise RuntimeError("torch is required to load the model")
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    vae = AutoencoderKLWan.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        model_dir, vae=vae, torch_dtype=torch_dtype
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    try:
        pipe.enable_model_cpu_offload()
    except Exception as e:  # pragma: no cover - accelerate may be absent
        print(f"[WARN] CPU offload unavailable: {e}")
    pipe.transformer.to("cuda")
    return pipe


def run_generation(pipe, params: argparse.Namespace, attn_name: str, attn_ctx):
    if torch is None:
        raise RuntimeError("torch is required to run generation")
    from diffusers.utils import export_to_video

    generator = None
    if params.seed >= 0:
        generator = torch.Generator("cuda").manual_seed(int(params.seed))

    kwargs: Dict = {
        "prompt": params.prompt,
        "negative_prompt": params.neg_prompt or None,
        "height": params.height,
        "width": params.width,
        "num_frames": params.frames,
        "num_inference_steps": params.steps,
        "guidance_scale": params.cfg,
        "num_videos_per_prompt": params.batch_size,
        "generator": generator,
        "output_type": "np",
    }
    if params.mode in {"i2v", "ti2v"} and params.image:
        kwargs["image"] = load_image(params.image, params.width, params.height)

    allowed = {
        "prompt",
        "negative_prompt",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "num_videos_per_prompt",
        "generator",
        "output_type",
        "image",
    }
    assert set(kwargs).issubset(allowed)

    outputs: List[str] = []
    base = str(int(time.time() * 1000))
    for bc in range(params.batch_count):
        start = time.time()
        with attn_ctx:
            result = pipe(**kwargs)
        elapsed = time.time() - start
        print(
            f"[INFO] Batch {bc}: {elapsed:.2f}s total, {elapsed / params.steps:.3f}s/step"
        )
        vids = getattr(result, "frames", getattr(result, "images", []))
        for bi, arr in enumerate(vids):
            if params.frames == 1:
                img = Image.fromarray(arr[0])
                fname = f"{base}_{bc:02d}_{bi:02d}.png"
                fpath = Path(params.outdir) / fname
                img.save(fpath)
            else:
                fname = f"{base}_{bc:02d}_{bi:02d}.mp4"
                fpath = Path(params.outdir) / fname
                export_to_video(arr, output_video_path=fpath, fps=params.fps)
            outputs.append(str(fpath))
    return outputs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["t2v", "i2v", "ti2v"], default="t2v")
    p.add_argument("--prompt", default="")
    p.add_argument("--neg_prompt", default="")
    p.add_argument("--sampler", default="unipc")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--batch_count", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--model_dir", default="")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn", choices=["auto", "flash", "sdpa", "math"], default="auto")
    p.add_argument("--image", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def validate(p: argparse.Namespace) -> None:
    p.width = snap32(p.width)
    p.height = snap32(p.height)
    if p.frames < 1:
        raise ValueError("frames must be >=1")
    if p.steps < 1:
        raise ValueError("steps must be >=1")
    if p.batch_count < 1 or p.batch_size < 1:
        raise ValueError("batch_count and batch_size must be >=1")
    if p.mode in {"t2v", "ti2v"} and not p.prompt.strip():
        raise ValueError("prompt required for text-to-video")
    if p.mode in {"i2v", "ti2v"} and not p.image:
        raise ValueError("image required for image-based modes")
    if p.outdir:
        Path(p.outdir).mkdir(parents=True, exist_ok=True)
    if not p.dry_run and not p.model_dir:
        raise ValueError("model_dir is required unless --dry-run")


def main() -> int:
    args = parse_args()
    cfg = vars(args).copy()
    try:
        validate(args)
    except Exception as e:
        outdir = Path(args.outdir or ".")
        outdir.mkdir(parents=True, exist_ok=True)
        sidecar = outdir / f"error_{int(time.time()*1000)}.json"
        err = {
            "error": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc(),
            "config": cfg,
        }
        save_sidecar(sidecar, err)
        print("[RESULT] FAIL GENERATION", json.dumps(err))
        return 1

    if args.dry_run:
        sidecar = Path(args.outdir) / f"dryrun_{int(time.time()*1000)}.json"
        data = {"ok": True, "dry_run": True, "config": cfg}
        save_sidecar(sidecar, data)
        print("[RESULT] OK", json.dumps(data))
        return 0

    try:
        log_vram("before load")
        pipe = load_pipeline(args.model_dir, args.dtype)
        log_vram("after load")
        attn_name, attn_ctx = attention_context(args.attn)
        print(f"[INFO] Attention backend: {attn_name}")
        outputs = run_generation(pipe, args, attn_name, attn_ctx)
        sidecar = Path(args.outdir) / f"result_{int(time.time()*1000)}.json"
        data = {"ok": True, "outputs": outputs, "config": cfg}
        save_sidecar(sidecar, data)
        print("[RESULT] OK", json.dumps(data))
        return 0
    except Exception as e:
        sidecar = Path(args.outdir) / f"error_{int(time.time()*1000)}.json"
        data = {
            "error": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc(),
            "config": cfg,
        }
        save_sidecar(sidecar, data)
        print("[RESULT] FAIL GENERATION", json.dumps(data))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
