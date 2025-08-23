import argparse
import gc
import inspect
import json
import multiprocessing as mp
import os
import sys
import time

from typing import Optional

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

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as e:
    raise RuntimeError("Transformers AutoModel import failed; check versions") from e


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
    p = os.environ.get("WAN22_MODEL_DIR")
    return p if p and os.path.isdir(p) else None

def build_pipe(model_dir: str, dtype_str: str = "bfloat16"):
    import torch
    from diffusers import AutoencoderKL

    # WanPipeline used to be exported from diffusers but newer versions only
    # expose it via trust_remote_code.  Try the explicit import first and fall
    # back to DiffusionPipeline with remote code if it isn't available.
    try:  # new diffusers versions (<0.30) drop WanPipeline from the top level
        from diffusers import WanPipeline as _Pipe
        pipe_kwargs = {}
    except Exception:
        from diffusers import DiffusionPipeline as _Pipe  # type: ignore
        pipe_kwargs = {"trust_remote_code": True}
        log("WanPipeline not found in diffusers; using DiffusionPipeline with trust_remote_code")

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

    # Keep VAE numerics in float32 for stability.  The WAN 2.2 checkpoints use a
    # custom VAE implementation that requires trusting the remote code to load
    # correctly.  Without `trust_remote_code=True` diffusers falls back to the
    # builtin AutoencoderKL class, which results in many missing weight errors
    # like `decoder.mid_block.attentions.0.to_q.weight`.
    vae = AutoencoderKL.from_pretrained(
        model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None,
        trust_remote_code=True,
    )
    text_encoder = AutoModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

    try:
        pipe = _Pipe.from_pretrained(
            model_dir,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            **pipe_kwargs,
        )
    except Exception as e:
        log(f"Failed to load pipeline: {e}")
        raise
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
            try:
                pipe.unet.to(memory_format=torch.channels_last)
            except Exception:
                pass
            try:
                pipe.text_encoder.to(memory_format=torch.channels_last)
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
    args = p.parse_args()

    init_worker(args.model_dir, args.dtype)
    res = generate(**vars(args))
    log(f"Generation completed: {res}")
    shutdown_worker()
    return 0

if __name__ == "__main__":
    sys.exit(main())
