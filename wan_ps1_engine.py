import argparse, json, os, sys, time, gc
import multiprocessing as mp
from typing import Optional

# ---- SDPA shim to ignore unexpected kwargs (e.g. enable_gqa) ----
try:
    import torch
    import torch.nn.functional as _F
    _orig_sdpa = _F.scaled_dot_product_attention
    def _sdpa_shim(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)
    _F.scaled_dot_product_attention = _sdpa_shim
except Exception:
    pass

# --- perf: tf32 + sdpa kernels ---
# global pipeline/worker state
_PIPE = None
_JOB_QUEUE = None
_WORKER = None

_sdpa_ctx = None
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
    try:
        from torch.nn.attention import sdpa_kernel
        _sdpa_ctx = sdpa_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
    except Exception:
        _sdpa_ctx = None
except Exception:
    _sdpa_ctx = None
# --- end perf ---

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
    p = os.environ.get("WAN22_MODEL_DIR")
    return p if p and os.path.isdir(p) else None

def build_pipe(model_dir: str, dtype_str: str = "bfloat16"):
    import torch
    from diffusers import WanPipeline, AutoModel
    torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

    # The VAE's config may contain stale keys (e.g. clip_output) that newer
    # diffusers versions warn about. Remove them before loading to keep the
    # console clean and avoid confusing users.
    cfg_path = os.path.join(model_dir, "vae", "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.pop("clip_output", None) is not None:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            log("Removed deprecated 'clip_output' from VAE config")
    except Exception as e:
        log(f"VAE config cleanup failed: {e}")

    # Keep VAE numerics in float32 for stability
    vae = AutoModel.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_dir, vae=vae, torch_dtype=torch_dtype)
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
    if n < 1: return 1
    rem = (n - 1) % 4
    if rem == 0: return n
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
    import imageio_ffmpeg as ffmpeg, numpy as np
    frames = iter(frames)
    first = np.asarray(next(frames))
    h, w = first.shape[:2]
    writer = ffmpeg.write_frames(outpath, size=(w, h), fps=fps)
    writer.send(None)
    try:
        writer.send(first)
        for frame in frames:
            writer.send(np.asarray(frame))
    finally:
        writer.close()

def _init_pipe(model_dir: Optional[str], dtype: str):
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

    log(f"Loading model: {model_dir}")
    pipe = build_pipe(model_dir, dtype)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            pipe.to("cuda")
            try: pipe.unet.to(memory_format=torch.channels_last)
            except Exception: pass
            try: pipe.text_encoder.to(memory_format=torch.channels_last)
            except Exception: pass
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log("Xformers memory-efficient attention enabled")
            except Exception:
                pass
            log("Moved pipeline to CUDA")
        except Exception as e:
            log(f"Failed to move pipe to CUDA: {e}")
            try:
                pipe.enable_model_cpu_offload()
                log("Falling back to CPU offload")
            except Exception as e2:
                log(f"CPU offload failed: {e2}")
                raise
    else:
        try:
            pipe.enable_model_cpu_offload()
            log("Falling back to CPU/offload mode")
        except Exception as e2:
            log(f"CPU offload failed: {e2}")
            raise

    # Optional memory tweaks
    try:
        pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
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

    result, last_err = None, None
    for with_cb in (True, False):
        try:
            kwargs = dict(common)
            if with_cb:
                kwargs["callback"] = _cb; kwargs["callback_steps"] = 1
            log(f"Generating… steps={steps} frames={frames} fps={fps}")
            def _run_pipe():
                if _sdpa_ctx is not None:
                    with torch.inference_mode(), _sdpa_ctx:
                        return pipe(**kwargs)
                else:
                    with torch.inference_mode():
                        return pipe(**kwargs)
            try:
                result = _run_pipe()
            except torch.cuda.OutOfMemoryError:
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
            last_err = e; continue

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

def _worker_loop(q, model_dir, dtype):
    try:
        pipe = _init_pipe(model_dir, dtype)
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

def init_worker(model_dir: Optional[str] = None, dtype: str = "bfloat16"):
    """Start background worker that holds the pipeline in memory."""
    global _WORKER, _JOB_QUEUE
    if _WORKER is not None:
        return
    _JOB_QUEUE = mp.Queue()
    _WORKER = mp.Process(target=_worker_loop, args=(_JOB_QUEUE, model_dir, dtype), daemon=True)
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
        pipe = _init_pipe(params.get("model_dir"), params.get("dtype", "bfloat16"))
        return _generate_with_pipe(pipe, params)

    rq = mp.Queue()
    _JOB_QUEUE.put((params, rq))
    result = rq.get()
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(result["error"])
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="t2v", choices=["t2v","i2v","ti2v"])
    p.add_argument("--prompt", default="");            p.add_argument("--neg_prompt", default="")
    p.add_argument("--sampler", default="unipc");      p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=7.0);  p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=24);     p.add_argument("--frames", type=int, default=49)
    p.add_argument("--width", type=int, default=1024); p.add_argument("--height", type=int, default=576)
    p.add_argument("--batch_count", type=int, default=1); p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--outdir", default="outputs");     p.add_argument("--model_dir", default="")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    init_worker(args.model_dir, args.dtype)
    res = generate(**vars(args))
    log(f"Generation completed: {res}")
    shutdown_worker()
    return 0

if __name__ == "__main__":
    sys.exit(main())
