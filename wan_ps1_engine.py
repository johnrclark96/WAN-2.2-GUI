#!/usr/bin/env python
"""CLI engine for WAN 2.2 with lazy imports and strict arg handling."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

torch: Any = None
Image: Any = None
export_to_video: Any = None
load_image: Any = None
WanPipeline: Any = None
AutoencoderKLWan: Any = None
UniPCMultistepScheduler: Any = None


def snap32(x: int) -> int:
    """Snap ``x`` down to the nearest multiple of 32."""
    return max(32, int(x) // 32 * 32)


def log_vram(label: str) -> None:
    """Print VRAM usage for debugging."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch optional in tests
        return
    if not torch.cuda.is_available():  # pragma: no cover - GPU may be absent
        return
    free, total = torch.cuda.mem_get_info()  # type: ignore[arg-type]
    used = (total - free) / 1024**2
    print(f"[INFO] {label} VRAM: {used:.0f} MiB used / {total/1024**2:.0f} MiB total")


def attention_context(pref: str):
    """Return (name, context manager) for the requested attention backend."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
    except Exception:  # pragma: no cover - torch absent
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


def save_sidecar(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def _lazy_diffusers():  # pragma: no cover - imported only in real runs
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
    from diffusers.utils import export_to_video, load_image

    return (
        WanPipeline,
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        export_to_video,
        load_image,
    )


def _lazy_utils(require_image: bool = False, force: bool = False):  # pragma: no cover
    global export_to_video, load_image, Image
    need_export = force or export_to_video is None
    need_load = force or load_image is None
    need_img = require_image and Image is None
    if not (need_export or need_load or need_img):
        return
    from diffusers.utils import export_to_video as _export_to_video
    load_image_module: Any = None
    if need_load:
        try:
            from diffusers.utils import load_image as _load_image_func
            load_image_module = _load_image_func
        except Exception:
            load_image_module = None
    ImageCls: Any = None
    if need_img:
        from PIL import Image as _ImageCls
        ImageCls = _ImageCls
    if need_export:
        export_to_video = _export_to_video
    if need_load and load_image_module is not None:
        load_image = load_image_module
    if need_img and ImageCls is not None:
        Image = ImageCls


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


def load_pipeline(model_dir: str, dtype: str):  # pragma: no cover - heavy
    global WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler, torch
    if WanPipeline is None:
        WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler, _, _ = _lazy_diffusers()
    if torch is None:
        import torch as _torch
        torch = _torch
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


def run_generation(
    pipe,
    params: argparse.Namespace,
    attn_name: str,
    attn_ctx,
) -> List[str]:
    global torch, Image, export_to_video, load_image
    need_img = params.frames == 1 or (
        params.mode in {"i2v", "ti2v"} and params.image
    )
    _lazy_utils(require_image=need_img, force=True)
    generator = None
    if params.seed >= 0 and torch is not None:
        generator = torch.Generator("cuda").manual_seed(int(params.seed))

    kwargs: Dict[str, Any] = {
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
        img = load_image(params.image).convert("RGB").resize(
            (params.width, params.height), Image.BICUBIC
        )
        kwargs["image"] = img

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
        print(f"[INFO] Batch {bc}: {elapsed:.2f}s total, {elapsed / params.steps:.3f}s/step")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["t2v", "i2v", "ti2v"], default="t2v")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--neg_prompt", default="")
    parser.add_argument("--sampler", default="unipc")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--batch_count", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--model_dir", default="")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--attn", choices=["auto", "flash", "sdpa", "math"], default="auto"
    )
    parser.add_argument("--image", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
        data = {"ok": True, "config": cfg}
        save_sidecar(sidecar, data)
        print("[RESULT] OK", json.dumps(data))
        return 0

    log_vram("before load")
    pipe = load_pipeline(args.model_dir, args.dtype)
    log_vram("after load")

    attn_name, attn_ctx = attention_context(args.attn)
    print(f"[INFO] Attention backend: {attn_name}")

    try:
        outputs = run_generation(
            pipe,
            args,
            attn_name,
            attn_ctx,
        )
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

    sidecar = Path(args.outdir) / f"result_{int(time.time()*1000)}.json"
    data = {"ok": True, "outputs": outputs, "config": cfg}
    save_sidecar(sidecar, data)
    print("[RESULT] OK", json.dumps(data))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
