import argparse
import gc
import inspect
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

from typing import Optional


def OK(tag: str, **m):
    print("[RESULT] OK", tag, json.dumps(m, default=str))
    sys.exit(0)


def FAIL(tag: str, **m):
    print("[RESULT] FAIL", tag, json.dumps(m, default=str))
    sys.exit(1)

# ---- SDPA helper to ignore unsupported kwargs (e.g. enable_gqa) ----
def _patch_sdpa_for_gqa():
    """Patch torch's SDPA variants to drop enable_gqa if unsupported."""
    try:
        import torch

        modules = []
        try:
            import torch.nn.functional as _F
            modules.append(_F)
        except Exception:
            pass
        try:
            import torch.nn.attention as _A
            modules.append(_A)
        except Exception:
            pass
        try:
            modules.append(torch._C._nn)
        except Exception:
            pass

        def _make_shim(fn):
            def _shim(*args, **kwargs):
                kwargs.pop("enable_gqa", None)
                return fn(*args, **kwargs)
            return _shim

        for m in modules:
            try:
                fn = getattr(m, "scaled_dot_product_attention", None)
                if not fn:
                    continue
                try:
                    sig = inspect.signature(fn)
                except (ValueError, TypeError):
                    sig = None
                if sig and "enable_gqa" in sig.parameters:
                    continue
                setattr(m, "scaled_dot_product_attention", _make_shim(fn))
            except Exception:
                pass
    except Exception:
        pass

_patch_sdpa_for_gqa()

# --- perf: tf32 kernels ---
# global pipeline/worker state
_PIPE = None
_JOB_QUEUE = None
_WORKER = None

try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
except Exception:
    pass
# --- end perf ---


def _detect_flash_attn() -> bool:
    """Return True if flash_attn is available and GPU is supported."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import flash_attn  # noqa: F401
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def _select_attn_ctx(pref: str):
    """Select attention backend and return (name, context)."""
    try:
        from torch.backends.cuda import sdp_kernel
    except Exception:
        return "sdpa", None

    use_flash = pref == "flash" or (pref == "auto" and _detect_flash_attn())
    if use_flash:
        try:
            return "flash", sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        except Exception:
            pass
    try:
        return "sdpa", sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
    except Exception:
        return "sdpa", None

def log(msg: str):
    print(f"[ps1-engine] {msg}", flush=True)

def detect_model_dir(root: str) -> Optional[str]:
    cands = [
        os.path.join(root, "models", "Wan2.2-TI2V-5B-Diffusers"),
        os.path.join(root, "models", "Wan2.2-T2V-14B-Diffusers"),
        os.path.join(root, "models", "Wan2.2-I2V-A14B-Diffusers"),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    env_p = os.environ.get("WAN22_MODEL_DIR")
    return env_p if env_p and os.path.isdir(env_p) else None


def validate_params(params: dict):
    """Validate key generation parameters."""
    errors = []
    if not isinstance(params.get("prompt", ""), str):
        errors.append(f"prompt invalid: {params.get('prompt')}")
    if not isinstance(params.get("neg_prompt", ""), str):
        errors.append(f"neg_prompt invalid: {params.get('neg_prompt')}")
    if int(params.get("steps", 0)) <= 0:
        errors.append(f"steps must be >0 (got {params.get('steps')})")
    if float(params.get("cfg", 0)) <= 0:
        errors.append(f"cfg must be >0 (got {params.get('cfg')})")
    for k in ("width", "height"):
        v = int(params.get(k, 0))
        if v <= 0 or v % 8 != 0:
            errors.append(f"{k} must be positive multiple of 8 (got {v})")
    if int(params.get("frames", 0)) <= 0:
        errors.append(f"frames must be >0 (got {params.get('frames')})")
    if int(params.get("fps", 0)) <= 0:
        errors.append(f"fps must be >0 (got {params.get('fps')})")
    sampler = str(params.get("sampler", "")).lower()
    allowed = {"unipc", "euler", "euler_a", "dpm"}
    if sampler not in allowed:
        errors.append(f"sampler must be one of {sorted(allowed)} (got {sampler})")
    if errors:
        raise ValueError("; ".join(errors))


def verify_vae_config(cfg_path: str) -> dict:
    """Ensure the VAE config describes the expected WAN video VAE."""
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise RuntimeError("Wrong VAE type (2D) — use WAN video VAE.")
    except Exception as e:
        raise RuntimeError(f"VAE config read failed: {e}") from e

    # Drop deprecated keys to avoid diffusers warnings.
    if cfg.pop("clip_output", None) is not None:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        log("Removed deprecated 'clip_output' from VAE config")

    required = ["base_dim", "z_dim", "scale_factor_spatial", "scale_factor_temporal"]
    if any(k not in cfg for k in required) or cfg.get("base_dim") != 160 or cfg.get("z_dim") != 48:
        raise RuntimeError("Wrong VAE type (2D) — use WAN video VAE.")

    log(
        "Detected WAN video VAE config: "
        f"base_dim={cfg['base_dim']}, z_dim={cfg['z_dim']}"
    )
    return cfg


def build_pipe(model_dir: str, dtype_str: str = "bfloat16", ignore_mismatch: bool = False):
    import torch
    from diffusers import DiffusionPipeline

    torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

    cfg_path = os.path.join(model_dir, "vae", "config.json")
    cfg = verify_vae_config(cfg_path)

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=ignore_mismatch,
        )
    except Exception as e:
        log(f"Failed to load pipeline: {e}")
        raise

    # Ensure the VAE runs in float32 for numerical stability and confirm it's 3D.
    try:
        pipe.vae.to(dtype=torch.float32)
    except Exception:
        pass

    latent_channels = getattr(pipe.vae.config, "latent_channels", "unknown")
    weight = pipe.vae.encoder.conv_in.weight
    rank = getattr(weight, "ndim", 0)
    if rank != 5:
        raise RuntimeError("Wrong VAE type (2D) — use WAN video VAE.")
    log(
        f"VAE {pipe.vae.__class__.__name__}: latent_channels={latent_channels}, "
        f"base_dim={cfg['base_dim']}, z_dim={cfg['z_dim']}, conv_in_rank={rank}D"
    )

    return pipe

def apply_wan_scheduler_fix(pipe, sampler: str, width: int, height: int):
    """Flow-aware schedulers for WAN 2.2:
       - euler/euler_a -> FlowMatchEulerDiscreteScheduler (stochastic for euler_a)
       - default (unipc) -> UniPCMultistepScheduler with flow_prediction/use_flow_sigmas/flow_shift
    """
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
    except ImportError as e:
        log(f"Scheduler imports failed: {e}")
        return

    flow_shift = 5.0 if max(width, height) >= 720 else 3.0
    s = (sampler or "unipc").lower()

    if s in ("euler", "euler_a"):
        try:
            sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        except (AttributeError, ValueError) as e:
            log(f"Using default FlowMatchEulerDiscreteScheduler: {e}")
            sched = FlowMatchEulerDiscreteScheduler()

        try:
            sched.register_to_config(shift=flow_shift)
        except AttributeError:
            try:
                setattr(sched, "shift", flow_shift)
            except (AttributeError, TypeError) as e:
                log(f"Failed to set shift attribute: {e}")
        except Exception as e:
            log(f"Failed to apply shift via register_to_config: {e}")

        if s == "euler_a":
            try:
                sched.register_to_config(stochastic_sampling=True)
            except AttributeError:
                try:
                    setattr(sched, "stochastic_sampling", True)
                except (AttributeError, TypeError) as e:
                    log(f"Failed to enable stochastic sampling: {e}")
            except Exception as e:
                log(f"Failed to enable stochastic sampling via register_to_config: {e}")

        pipe.scheduler = sched
        log(f"Scheduler set to {s} (FlowMatchEuler, shift={flow_shift})")
    else:
        try:
            sched = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        except (AttributeError, ValueError) as e:
            log(f"Using default UniPCMultistepScheduler: {e}")
            sched = UniPCMultistepScheduler()

        for k, v in (("prediction_type", "flow_prediction"),
                     ("use_flow_sigmas", True),
                     ("flow_shift", flow_shift)):
            try:
                sched.register_to_config(**{k: v})
            except AttributeError:
                try:
                    setattr(sched, k, v)
                except (AttributeError, TypeError) as e:
                    log(f"Failed to set {k}: {e}")
            except Exception as e:
                log(f"Failed to configure {k} via register_to_config: {e}")

        pipe.scheduler = sched
        log("Scheduler set to unipc (flow_prediction, use_flow_sigmas=True, shift={flow_shift})".format(flow_shift=flow_shift))

def round_frames(n: int) -> int:
    if n < 1:
        return 1
    rem = (n - 1) % 4
    if rem == 0:
        return n
    down = n - rem
    up = down + 4
    return up if (n - down) >= (up - n) else down

def normalize_resolution(pipe, w: int, h: int):
    """Snap H/W to the nearest multiple allowed by the model."""
    try:
        mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        w0, h0 = int(w), int(h)
        w = (w0 // mod) * mod
        h = (h0 // mod) * mod
        if w != w0 or h != h0:
            log(f"Snapped resolution from {w0}x{h0} to {w}x{h} (mod={mod}).")
    except Exception as e:
        log(f"Resolution snap failed: {e}")
    return w, h

def save_video(frames, fps: int, outpath: str):
    """Stream frames to an ffmpeg writer without manual cache clearing."""
    try:
        import imageio_ffmpeg as ffmpeg
        import numpy as np
    except Exception as e:
        log(f"Video writer imports failed: {e}")
        raise

    frames = iter(frames)
    try:
        first = np.asarray(next(frames))
    except StopIteration:
        raise RuntimeError("No frames to save")

    h, w = first.shape[:2]
    writer = ffmpeg.write_frames(outpath, size=(w, h), fps=fps)
    writer.send(None)
    try:
        writer.send(first)
        for frame in frames:
            writer.send(np.asarray(frame))
    finally:
        writer.close()

def _init_pipe(
    model_dir: Optional[str],
    dtype: str,
    ignore_mismatch: bool = False,
    compile_mode: str = "off",
):
    """Load and cache the diffusers pipeline."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    try:
        import torch
    except Exception as e:
        log(f"PyTorch import failed: {e}")
        raise

    root = os.path.dirname(os.path.abspath(__file__))
    model_dir = model_dir or detect_model_dir(root)
    if not model_dir:
        raise RuntimeError("Model directory not found. Set --model_dir or WAN22_MODEL_DIR.")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            free0, total0 = torch.cuda.mem_get_info()
            log(
                f"VRAM before load: {free0 // (1024**2)} MiB free / {total0 // (1024**2)} MiB total"
            )
        except Exception:
            pass

    log(f"Loading model: {model_dir}")
    pipe = build_pipe(model_dir, dtype, ignore_mismatch)

    if compile_mode != "off":
        try:
            pipe.unet = torch.compile(pipe.unet, mode=compile_mode)
            log(f"Compiled UNet with torch.compile (mode={compile_mode})")
        except Exception as e:
            log(f"torch.compile failed: {e}")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            pipe.to("cuda")
            try:
                pipe.unet.to(memory_format=torch.channels_last)
            except Exception:
                pass
            try:
                pipe.text_encoder.to(memory_format=torch.channels_last)
            except Exception:
                pass
            log("Moved pipeline to CUDA")
            log(f"Device: cuda, dtype: {pipe.unet.dtype}")
        except Exception as e:
            log(f"Failed to move pipe to CUDA: {e}")
            try:
                pipe.enable_model_cpu_offload()
                log("Falling back to CPU offload")
            except Exception as e2:
                log(f"CPU offload failed: {e2}")
                raise
        try:
            free1, total1 = torch.cuda.mem_get_info()
            log(
                f"VRAM after load: {free1 // (1024**2)} MiB free / {total1 // (1024**2)} MiB total"
            )
        except Exception:
            pass
    else:
        try:
            pipe.enable_model_cpu_offload()
            log("Falling back to CPU/offload mode")
        except Exception as e2:
            log(f"CPU offload failed: {e2}")
            raise
        log(f"Device: cpu, dtype: {pipe.unet.dtype}")

    # Optional memory tweaks
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    _PIPE = pipe
    return _PIPE

def _generate_with_pipe(pipe, params: dict):
    """Generate video using an existing pipeline instance."""
    import torch

    validate_params(params)

    mode = params.get("mode", "t2v")
    prompt = params.get("prompt", "")
    neg_prompt = params.get("neg_prompt", "")
    sampler = params.get("sampler", "unipc")
    steps = int(params.get("steps", 20))
    cfg = float(params.get("cfg", 7.0))
    seed = int(params.get("seed", 0))
    fps = int(params.get("fps", 24))
    frames_req = int(params.get("frames", 49))
    width_req = int(params.get("width", 1024))
    height_req = int(params.get("height", 576))
    outdir = params.get("outdir", "outputs")

    # Seed (optional)
    try:
        g = torch.Generator(device="cuda")
        if seed:
            g.manual_seed(seed)
    except Exception:
        g = None

    # Snap resolution to model granularity
    width, height = normalize_resolution(pipe, width_req, height_req)

    apply_wan_scheduler_fix(pipe, sampler, width, height)

    frames = round_frames(frames_req)
    if frames != frames_req:
        log("`num_frames - 1` has to be divisible by 4. Rounding to the nearest number.")

    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base  = f"WAN22_{mode}_{stamp}"
    json_path = os.path.join(outdir, base + ".json")
    mp4_path  = os.path.join(outdir, base + ".mp4")

    meta = {
        "mode": mode, "prompt": prompt, "neg_prompt": neg_prompt,
        "sampler": sampler, "steps": steps, "cfg": cfg, "seed": seed,
        "fps": fps, "frames": frames, "width": width, "height": height,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote: {json_path}")

    steps_total = max(1, steps)
    def _cb(step, timestep, latents):
        cur = int(step) + 1
        pct = int(min(100, max(0, round(cur * 100 / steps_total))))
        msg = {"event": "progress", "step": cur, "total": steps_total, "percent": pct}
        print(json.dumps(msg), flush=True)

    common = dict(
        prompt=prompt, negative_prompt=neg_prompt or None,
        width=width, height=height, num_inference_steps=steps,
        guidance_scale=cfg, num_frames=frames, generator=g, output_type="np",
    )
    attn_pref = (params.get("attn", "auto") or "auto").lower()
    backend, attn_ctx = _select_attn_ctx(attn_pref)
    if attn_pref == "flash" and backend != "flash":
        log("flash_attn requested but unavailable; using SDPA")
    log(f"Attention backend: {backend}")

    mesh = params.get("mesh", "off")
    if mesh != "off":
        log(f"Mesh strategy: {mesh}")

    result, last_err = None, None
    for with_cb in (True, False):
        try:
            kwargs = dict(common)
            if with_cb:
                kwargs["callback"] = _cb
                kwargs["callback_steps"] = 1
            log(f"Generating… steps={steps} frames={frames} fps={fps}")

            def _run_pipe(inner_pipe=pipe):
                if attn_ctx is not None:
                    with torch.inference_mode(), attn_ctx:
                        return inner_pipe(**kwargs)
                with torch.inference_mode():
                    return inner_pipe(**kwargs)

            try:
                result = _run_pipe()
            except torch.cuda.OutOfMemoryError:
                try:
                    free, total = torch.cuda.mem_get_info()
                    log(
                        "CUDA OOM: free={free} total={total}. Suggest smaller width/height/frames".format(
                            free=free // (1024**2), total=total // (1024**2)
                        )
                    )
                except Exception:
                    log("CUDA out of memory. Falling back to sequential CPU offload and retrying.")
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception as e2:
                    log(f"Sequential CPU offload failed: {e2}; switching to CPU")
                    try:
                        pipe.to("cpu")
                    except Exception as e3:
                        log(f"CPU move failed: {e3}")
                        raise
                result = _run_pipe()
            break
        except TypeError as e:
            last_err = e
            continue

    if result is None:
        raise RuntimeError(f"Pipeline call failed: {last_err}")

    frames_out = None
    if isinstance(result, dict):
        frames_out = result.get("frames")
        if frames_out is None:
            frames_out = result.get("images")
    else:
        frames_out = getattr(result, "frames", None)
        if frames_out is None:
            frames_out = getattr(result, "images", None)
    if frames_out is None:
        raise RuntimeError("No frames returned from pipeline")

    if frames <= 1 or len(frames_out) <= 1:
        try:
            from PIL import Image
            img_path = os.path.join(outdir, base + ".png")
            Image.fromarray(frames_out[0]).save(img_path)
            log(f"wrote: {img_path}")
            print(json.dumps({"event": "done", "image": img_path}), flush=True)
            del result, frames_out, pipe
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return {"json": json_path, "image": img_path}
        except Exception as e:
            log(f"Failed to save image: {e}")
            raise

    def _frame_gen(frames_list):
        for i, frame in enumerate(frames_list):
            yield frame
            frames_list[i] = None

    save_video(_frame_gen(frames_out), fps, mp4_path)
    log(f"wrote: {mp4_path}")

    print(json.dumps({"event": "done", "video": mp4_path}), flush=True)
    del result, frames_out, pipe
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {"json": json_path, "mp4": mp4_path}


def _worker_loop(q, model_dir, dtype, ignore_mismatch, compile_mode):
    try:
        pipe = _init_pipe(model_dir, dtype, ignore_mismatch, compile_mode)
    except Exception as e:
        log(f"Worker failed to initialize pipeline: {e}")
        return
    while True:
        task = q.get()
        if task is None:
            break
        params, rq = task
        try:
            res = _generate_with_pipe(pipe, params)
            rq.put(res)
        except Exception as e:
            rq.put({"error": str(e)})

def init_worker(
    model_dir: Optional[str] = None,
    dtype: str = "bfloat16",
    ignore_mismatch: bool = False,
    compile_mode: str = "off",
):
    """Start background worker that holds the pipeline in memory."""
    global _WORKER, _JOB_QUEUE
    if _WORKER is not None:
        return
    _JOB_QUEUE = mp.Queue()
    _WORKER = mp.Process(
        target=_worker_loop,
        args=(_JOB_QUEUE, model_dir, dtype, ignore_mismatch, compile_mode),
        daemon=True,
    )
    _WORKER.start()

def shutdown_worker():
    global _WORKER, _JOB_QUEUE
    if _WORKER is None:
        return
    _JOB_QUEUE.put(None)
    _WORKER.join()
    _WORKER = None
    _JOB_QUEUE = None

def generate(**params):
    """Public API to generate a video. Blocks until job completion."""
    if _JOB_QUEUE is None:
        pipe = _init_pipe(
            params.get("model_dir"),
            params.get("dtype", "bfloat16"),
            params.get("ignore_mismatch", False),
            params.get("compile", "off"),
        )
        return _generate_with_pipe(pipe, params)

    # On Windows, multiprocessing uses the 'spawn' start method which cannot
    # pickle ``mp.Queue`` instances after process start.  Passing a standard
    # Queue through another queue (as we do for the response channel) raises
    # ``RuntimeError: Queue objects should only be shared between processes
    # through inheritance``.  Using a ``multiprocessing.Manager`` provides a
    # proxy object that *can* be safely serialized and shared.
    mgr = mp.Manager()
    try:
        rq = mgr.Queue()
        _JOB_QUEUE.put((params, rq))
        result = rq.get()
    finally:
        mgr.shutdown()

    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(result["error"])
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="t2v", choices=["t2v", "i2v", "ti2v"])
    p.add_argument("--prompt", default="")
    p.add_argument("--neg_prompt", default="")
    p.add_argument("--sampler", default="unipc")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--batch_count", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--model_dir", default="")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="auto", choices=["auto", "flash", "sdpa"])
    p.add_argument("--mesh", choices=["off", "grid", "tile"], default="off")
    p.add_argument(
        "--compile",
        choices=["off", "default", "reduce-overhead"],
        default="off",
    )
    p.add_argument("--ignore-mismatch", action="store_true", dest="ignore_mismatch")
    p.add_argument("--dry-run", "--no-model", action="store_true", dest="dry_run")
    p.add_argument("--print-config", action="store_true", dest="print_config")
    args = p.parse_args()

    params = vars(args).copy()
    try:
        validate_params(params)
    except Exception as e:
        FAIL("ARGS", error=str(e), params=params)

    if args.print_config:
        print(json.dumps(params, indent=2))
        OK("PRINT_CONFIG")

    if args.dry_run:
        try:
            root = os.path.dirname(os.path.abspath(__file__))
            model_dir = args.model_dir or detect_model_dir(root)
            if not model_dir:
                FAIL("DRY_RUN", error="Model directory not found", params=params)
            model_dir = os.path.abspath(model_dir)
            log(f"Model dir: {model_dir}")
            cfg_path = os.path.join(model_dir, "vae", "config.json")
            verify_vae_config(cfg_path)
            subfolders = ["vae", "unet", "text_encoder", "scheduler"]
            for sf in subfolders:
                log(f"Subfolder: {sf} -> {os.path.join(model_dir, sf)}")
            backend, _ = _select_attn_ctx(args.attn)
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "unknown"
            log(f"Attention backend: {backend}")
            log(f"Mesh: {args.mesh}")
            log(f"Compile: {args.compile}")
            log(f"Device: {device}, dtype: {args.dtype}")
            OK(
                "DRY_RUN",
                backend=backend,
                mesh=args.mesh,
                compile=args.compile,
                device=device,
                dtype=args.dtype,
                model_dir=model_dir,
                subfolders=subfolders,
            )
        except Exception as e:
            FAIL("DRY_RUN", error=str(e), tb=traceback.format_exc(), params=params)

    try:
        init_worker(args.model_dir, args.dtype, args.ignore_mismatch, args.compile)
        params.pop("dry_run", None)
        res = generate(**params)
        log(f"Generation completed: {res}")
        OK("GENERATION", result=res)
    except Exception as e:
        FAIL("GENERATION", error=str(e), tb=traceback.format_exc(), config=params)
    finally:
        shutdown_worker()
    return 0

if __name__ == "__main__":
    sys.exit(main())
