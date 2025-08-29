#!/usr/bin/env python
"""CLI engine for WAN 2.2 with lazy imports and strict arg handling."""

from __future__ import annotations

import argparse
import json
import time
import traceback
import os
from contextlib import nullcontext
from pathlib import Path
from core.paths import OUTPUT_DIR, MODELS_DIR
from typing import Any, Dict, List, IO

# ---------------- persistent logging helpers ----------------

_RUN_TS = time.strftime("%Y%m%d_%H%M%S")

def _init_log_dirs() -> tuple[Path, Path]:
    """Pick writable log dirs across environments (CI-friendly).
    Order: $WAN_LOG_ROOT > D:/wan22 (if exists) > CWD. Never raise at import-time.
    """
    candidates: list[Path] = []
    env_root = os.getenv("WAN_LOG_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    d_base = Path("D:/wan22")
    if d_base.exists():
        candidates.append(d_base)
    candidates.append(Path.cwd())
    for base in candidates:
        try:
            (base / "logs").mkdir(parents=True, exist_ok=True)
            (base / "json").mkdir(parents=True, exist_ok=True)
            return base / "logs", base / "json"
        except Exception:
            continue
    # last resort: CWD
    return Path.cwd() / "logs", Path.cwd() / "json"

_LOG_DIR_TXT, _LOG_DIR_JSON = _init_log_dirs()
_LOG_TXT = _LOG_DIR_TXT / f"wan_run_{_RUN_TS}.log"
_LOG_JSONL = _LOG_DIR_JSON / f"wan_run_{_RUN_TS}.jsonl"

def _now() -> float:
    return time.time()

def log(msg: str, stage: str = "info", **extra) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [{stage}] {msg}"
    print(line, flush=True)
    try:
        with _LOG_TXT.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    payload = {"ts": _now(), "stage": stage, "msg": msg}
    if extra:
        payload["extra"] = extra
    try:
        with _LOG_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------------- lazy/optional globals ----------------
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
        import torch as _t
    except Exception:  # pragma: no cover - torch optional in tests
        return
    if not _t.cuda.is_available():  # pragma: no cover - GPU may be absent
        return
    free, total = _t.cuda.mem_get_info()  # type: ignore[arg-type]
    used = (total - free) / 1024**2
    print(f"[INFO] {label} VRAM: {used:.0f} MiB used / {total/1024**2:.0f} MiB total")


def attention_context(pref: str, pipe: Any | None = None):
    """Return (name, context manager) for the requested attention backend."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
    except Exception:  # pragma: no cover - torch absent
        return "sdpa", nullcontext()

    name = "sdpa"
    ctx = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    if pref in ("auto", "flash-attn"):
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                name = "flash-attn"
                ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        except Exception as e:
            if pref == "flash-attn":
                print(f"[WARN] FA3 requested but unavailable ({e}); using sdpa")
    if pref == "flash-attn-ext" and pipe is not None:
        try:
            from diffusers.models.attention_processor import FlashAttention2Processor
            pipe.transformer.set_attn_processor(FlashAttention2Processor())
            name = "flash-attn-ext"
            ctx = nullcontext()
        except Exception as e:
            print(f"[WARN] flash-attn-ext requested but unavailable ({e}); using sdpa")
    return name, ctx


def _dtype_fixup(dtype: str) -> str:
    """Auto-fallback bf16 -> fp16 when bf16 isn't ideal/supported on this GPU."""
    try:
        import torch as _t
        if dtype.lower() == "bf16" and _t.cuda.is_available():
            major, _ = _t.cuda.get_device_capability()  # type: ignore[attr-defined]
            if major < 8:
                return "fp16"
    except Exception:
        pass
    return dtype


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
    if p.mode == "t2i":
        p.frames = 1
    if (p.width, p.height) == (1280, 720):
        p.height = 704
    elif (p.width, p.height) == (720, 1280):
        p.width = 704
    p.width = snap32(p.width)
    p.height = snap32(p.height)
    if p.frames < 1:
        p.frames = 1
    if (p.frames - 1) % 4 != 0:
        p.frames = (p.frames - 1) // 4 * 4 + 1
    if p.steps < 1:
        p.steps = 1
    if p.batch_count < 1 or p.batch_size < 1:
        raise ValueError("batch_count and batch_size must be >=1")
    if not p.prompt.strip():
        raise ValueError("prompt required for generation")
    if p.mode == "t2v" and p.image:
        if not Path(p.image).exists():
            raise ValueError(f"image not found: {p.image}")
    if p.frames > 64:
        print("[WARN] frames > 64 may exceed 16 GB VRAM")
    if p.width > 1280 or p.height > 704:
        print("[WARN] resolution > 1280x704 may exceed 16 GB VRAM")
    if p.outdir:
        Path(p.outdir).mkdir(parents=True, exist_ok=True)
    if not p.dry_run and not p.model_dir:
        raise ValueError("model_dir is required unless --dry-run")


def load_pipeline(model_dir: str, dtype: str):  # pragma: no cover - heavy
    """Optimized, quality-neutral loader for WAN pipeline."""
    global WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler, torch
    if WanPipeline is None:
        WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler, _, _ = _lazy_diffusers()
    if torch is None:
        import torch as _torch
        torch = _torch

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    t0 = _now()
    log(f"Loading model from: {model_dir}", stage="load")
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch_dtype)
    pipe = WanPipeline.from_pretrained(model_dir, vae=vae, torch_dtype=torch_dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Prefer memory layouts that speed Conv3d (no fidelity change)
    try:
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last_3d)
        log("VAE memory_format=channels_last_3d", stage="opt")
    except Exception:
        try:
            pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
            log("VAE memory_format=channels_last", stage="opt")
        except Exception:
            pass

    # Diffusers memory helpers (no output change)
    try:
        pipe.enable_attention_slicing()
        log("Attention slicing enabled", stage="opt")
    except Exception:
        log("Attention slicing unavailable", stage="warn")
    for _fn in (getattr(pipe.vae, "enable_slicing", None), getattr(pipe.vae, "enable_tiling", None)):
        try:
            if callable(_fn):
                _fn()
                log(f"VAE {_fn.__name__} enabled", stage="opt")
        except Exception:
            log("VAE slicing/tiling unavailable", stage="warn")

    # Steadier than model_cpu_offload; do not also call pipe.to('cuda')
    try:
        pipe.enable_sequential_cpu_offload()
        log("Sequential CPU offload enabled", stage="opt")
    except Exception as e:
        log(f"Sequential CPU offload unavailable: {e}", stage="warn")

    # Optional: compile VAE decoder for small overhead win (same math)
    import platform
    has_triton = True
    try:
        import triton  # type: ignore  # noqa: F401
    except Exception:
        has_triton = False
    if os.getenv("WAN_FORCE_COMPILE", "0") == "1" and has_triton and platform.system() != "Windows":
        try:
            pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="reduce-overhead", fullgraph=False)
            log("torch.compile for VAE decoder", stage="opt")
        except Exception as e:
            log(f"torch.compile skipped: {e}", stage="warn")
    else:
        log("torch.compile disabled (no Triton/Windows or WAN_FORCE_COMPILE!=1)", stage="opt")

    log(f"Pipeline loaded in {(_now()-t0):.2f}s (dtype={dtype})", stage="load")
    return pipe


def run_generation(
    pipe,
    params: argparse.Namespace,
    attn_name: str,
    attn_ctx,
) -> List[str]:
    """Generate and save, with persistent stage logs and fast encoding."""
    global torch, Image, export_to_video, load_image
    need_img = params.frames == 1 or bool(params.image)
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
    if params.image:
        _Resampling = getattr(Image, "Resampling", Image)
        _BICUBIC = getattr(_Resampling, "BICUBIC", getattr(Image, "BICUBIC", 3))
        img = load_image(params.image).convert("RGB").resize((params.width, params.height), _BICUBIC)
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

    # Announce decode start right after last denoise step if supported
    try:
        import inspect as _inspect

        def _last_step_cb(*cb_args, **cb_kwargs):
            """
            Compatible with WAN/Diffusers variants that call either:
              (pipe, i, t, callback_kwargs)  OR  (i, t, callback_kwargs)
            Must return a dict (callback_kwargs) for the pipeline to .pop() from.
            """
            # 1) Extract callback_kwargs dict
            cb_dict = None
            if "callback_kwargs" in cb_kwargs and isinstance(cb_kwargs["callback_kwargs"], dict):
                cb_dict = cb_kwargs["callback_kwargs"]
            elif cb_args and isinstance(cb_args[-1], dict):
                cb_dict = cb_args[-1]
            if cb_dict is None:
                cb_dict = {}

            # 2) Extract step index
            step = cb_kwargs.get("i")
            if not isinstance(step, int):
                if len(cb_args) >= 2 and isinstance(cb_args[1], int):
                    step = cb_args[1]
                elif len(cb_args) >= 1 and isinstance(cb_args[0], int):
                    step = cb_args[0]

            # 3) Log on last step (don't ever break the pipeline)
            try:
                if step is not None and int(step) == int(params.steps) - 1:
                    log("Denoising finished. Starting VAE decode...", stage="decode")
            except Exception:
                pass

            # Always return the dict the pipeline expects
            return cb_dict

        if "callback_on_step_end" in _inspect.signature(pipe.__call__).parameters:
            kwargs["callback_on_step_end"] = _last_step_cb
    except Exception:
        pass

    outputs: List[str] = []
    base = str(int(time.time() * 1000))

    # helper: check NVENC availability lazily
    def _nvenc_ok(codec: str) -> bool:
        import subprocess as _sp
        cache = getattr(_nvenc_ok, '_CACHE', {})
        cached = cache.get(codec)
        if cached is not None:
            return cached
        try:
            p = _sp.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, timeout=5)
            ok = (f'{codec}_nvenc' in p.stdout)
        except Exception:
            ok = False
        cache[codec] = ok
        setattr(_nvenc_ok, '_CACHE', cache)
        return ok

    # encoding defaults (safe if args lack these attrs)
    encoder = getattr(params, "encoder", "auto")      # auto|nvenc|cpu
    codec   = getattr(params, "codec",   "h264")      # h264|hevc
    mode    = getattr(params, "encode_mode", "delivery")  # delivery|lossless
    preset  = getattr(params, "nvenc_preset", "p4")   # p1..p7
    cq      = str(getattr(params, "nvenc_cq", 18))     # NVENC delivery quality

    for bc in range(params.batch_count):
        with attn_ctx:
            # Free space before decode inside __call__
            try:
                if hasattr(pipe, "transformer"):
                    pipe.transformer.to("cpu")
                    log("Transformer → CPU before decode", stage="opt")
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            start = time.time()
            log(f"Batch {bc} starting…", stage="gen")
            im = getattr(torch, "inference_mode", None)
            _ctx = im() if callable(im) else nullcontext()
            with _ctx:
                result = pipe(**kwargs)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            elapsed = time.time() - start
            print(f"[INFO] Batch {bc}: {elapsed:.2f}s total, {elapsed / max(1, params.steps):.3f}s/step")
            log(f"Batch {bc} finished in {elapsed:.2f}s", stage="gen")

        vids = getattr(result, "frames", getattr(result, "images", []))
        for bi, arr in enumerate(vids):
            if params.frames == 1:
                img = Image.fromarray(arr[0])
                fname = f"{base}_{bc:02d}_{bi:02d}.png"
                fpath = Path(params.outdir) / fname
                img.save(fpath)
                log(f"Saved image → {fpath}", stage="save")
            else:
                # Prefer NVENC when possible; verify codec + channels first
                fname = f"{base}_{bc:02d}_{bi:02d}.mp4"
                fpath = Path(params.outdir) / fname
                use_nvenc = False
                if (encoder in ("auto", "nvenc")) and hasattr(arr, "shape"):
                    try:
                        T, H, W, C = arr.shape  # type: ignore[attr-defined]
                        use_nvenc = _nvenc_ok(codec) and (C in (3, 4))
                    except Exception:
                        use_nvenc = False
                # Compute T/H/W defensively for logging and CPU fallback
                T = arr.shape[0] if hasattr(arr, "shape") else len(arr)
                H = arr.shape[1] if hasattr(arr, "shape") else (len(arr[0]) if T and hasattr(arr[0], "__len__") else 0)
                W = arr.shape[2] if hasattr(arr, "shape") else (len(arr[0][0]) if T and hasattr(arr[0], "__len__") and hasattr(arr[0][0], "__len__") else 0)
                log(f"Encoding video ({T} frames @ {params.fps} fps) → {fpath}", stage="encode")
                if use_nvenc:
                    # If frames are RGBA, drop alpha for ffmpeg rgb24 input
                    if getattr(arr, "shape", None) and arr.shape[-1] == 4:  # type: ignore[attr-defined]
                        arr = arr[..., :3]
                    import subprocess as _sp
                    enc = f"{codec}_nvenc"
                    if mode == "lossless":
                        ff = [
                            "ffmpeg", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(params.fps), "-i", "-",
                            "-c:v", enc, "-tune", "lossless", "-rc", "constqp", "-qp", "0",
                            "-preset", preset, "-profile:v", "high444p", "-pix_fmt", "yuv444p",
                            str(fpath),
                        ]
                        log(f"NVENC lossless ({enc}, preset {preset})", stage="encode")
                    else:
                        gop = max(1, int(params.fps * 2))
                        ff = [
                            "ffmpeg", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(params.fps), "-i", "-",
                            "-c:v", enc, "-preset", preset, "-rc", "vbr_hq", "-cq", cq,
                            "-bf", "2", "-g", str(gop), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(fpath),
                        ]
                        log(f"NVENC delivery ({enc}, preset {preset}, cq {cq})", stage="encode")
                    t1 = _now()
                    try:
                        proc = _sp.Popen(ff, stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE)
                        stdin: IO[bytes] | None = proc.stdin
                        if stdin is None:
                            raise RuntimeError("FFmpeg Popen.stdin is None despite PIPE")
                        for i in range(T):
                            stdin.write(arr[i].tobytes())
                            if (i % max(1, T // 20) == 0) or (i == T - 1):
                                pct = int((i + 1) * 100 / T)
                                log(f"encode progress {i+1}/{T} ({pct}%)", stage="encode")
                        stdin.close()
                        out, err = proc.communicate()
                        if proc.returncode != 0:
                            raise RuntimeError(err.decode("utf-8", errors="ignore"))
                        log(f"Encode finished in {(_now()-t1):.1f}s", stage="encode")
                    except Exception as e:
                        log(f"FFmpeg/NVENC failed: {e}. Falling back to CPU export_to_video", stage="warn")
                        export_to_video(arr, output_video_path=fpath, fps=params.fps)
                else:
                    # CPU path via Diffusers helper (also used when arr is a plain list in tests)
                    log("Encoding via export_to_video (CPU path / non-array input)", stage="encode")
                    export_to_video(arr, output_video_path=fpath, fps=params.fps)

            outputs.append(str(fpath))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["t2v", "t2i"], default="t2v")
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
    parser.add_argument("--outdir", default=OUTPUT_DIR.as_posix())
    parser.add_argument(
        "--model_dir",
        default=(MODELS_DIR / "Wan2.2-TI2V-5B-Diffusers").as_posix(),
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument(
        "--attn", choices=["auto", "sdpa", "flash-attn", "flash-attn-ext"], default="auto"
    )
    parser.add_argument("--image", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        payload = {
            "mode": args.mode,
            "prompt": args.prompt,
            "frames": int(args.frames),
            "width": int(args.width),
            "height": int(args.height),
        }
        print("[WAN shim] Dry run: " + json.dumps(payload, separators=(",", ":")))
        print("[RESULT] OK " + json.dumps(payload, separators=(",", ":")))
        return 0

    args.dtype = _dtype_fixup(args.dtype)
    if args.dtype.lower() != "bf16":
        print(f"[opt] dtype auto-adjusted to {args.dtype} (bf16 not ideal on this GPU)")

    cfg = vars(args).copy()

    try:
        validate(args)
    except Exception as e:
        outdir = Path(args.outdir or OUTPUT_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        sidecar = outdir / f"error_{int(time.time()*1000)}.json"
        err = {
            "error": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc(),
            "config": cfg,
        }
        save_sidecar(sidecar, err)
        print(f"[RESULT] FAIL VALIDATION {err['error']}")
        return 1

    if args.mode == "t2i":
        from core import wan_image
        gen = wan_image.generate_image_wan
    else:
        from core import wan_video
        gen = wan_video.generate_video_wan

    try:
        outputs, attn_used = gen(args)
        cfg["attention_backend"] = attn_used
    except Exception as e:
        sidecar = Path(args.outdir) / f"error_{int(time.time()*1000)}.json"
        gen_err: Dict[str, Any] = {
            "error": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc(),
            "config": cfg,
        }
        save_sidecar(sidecar, gen_err)
        print(f"[RESULT] FAIL GENERATION {gen_err['error']}")
        return 1

    for out in outputs[:1]:
        print(f"[OUTPUT] {Path(out).resolve()}")

    sidecar = Path(args.outdir) / f"result_{int(time.time()*1000)}.json"
    result_data: Dict[str, Any] = {"ok": True, "outputs": outputs, "config": cfg}
    save_sidecar(sidecar, result_data)
    print("[RESULT] OK done")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
