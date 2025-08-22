import argparse, json, os, sys, time
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

# --- perf: tf32 + sdp kernels ---
try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
    except Exception:
        pass
except Exception:
    pass
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
    except Exception as e:
        log(f"Scheduler imports failed: {e}"); return
    flow_shift = 5.0 if max(width, height) >= 720 else 3.0
    s = (sampler or "unipc").lower()
    try:
        if s in ("euler", "euler_a"):
            try:
                sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            except Exception:
                sched = FlowMatchEulerDiscreteScheduler()
            # apply shift if supported
            try:
                sched.register_to_config(shift=flow_shift)
            except Exception:
                try: setattr(sched, "shift", flow_shift)
                except Exception: pass
            if s == "euler_a":
                try:
                    sched.register_to_config(stochastic_sampling=True)
                except Exception:
                    try: setattr(sched, "stochastic_sampling", True)
                    except Exception: pass
            pipe.scheduler = sched
            log(f"Scheduler set to {s} (FlowMatchEuler, shift={flow_shift})")
        else:
            try:
                sched = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            except Exception:
                sched = UniPCMultistepScheduler()
            for k, v in (("prediction_type","flow_prediction"),
                         ("use_flow_sigmas", True),
                         ("flow_shift", flow_shift)):
                try:
                    sched.register_to_config(**{k: v})
                except Exception:
                    try: setattr(sched, k, v)
                    except Exception: pass
            pipe.scheduler = sched
            log(f"Scheduler set to unipc (flow_prediction, use_flow_sigmas=True, shift={flow_shift})")
    except Exception as e:
        log(f"Scheduler override skipped: {e}")

def round_frames(n: int) -> int:
    if n < 1: return 1
    rem = (n - 1) % 4
    if rem == 0: return n
    down = n - rem
    up = down + 4
    return up if (n - down) >= (up - n) else down

def normalize_resolution(model_dir: str, w: int, h: int):
    """For TI2V-5B, map 1280x720 to 1280x704 (official run-size)."""
    name = (model_dir or "").lower()
    if "ti2v-5b" in name and w == 1280 and h == 720:
        log("TI2V-5B: snapping 1280x720 to 1280x704 (official run-size).")
        return 1280, 704
    return w, h

def save_video(frames, fps: int, outpath: str):
    import numpy as np
    from diffusers.utils import export_to_video
    arr = np.asarray(frames)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    export_to_video(arr, outpath, fps=fps)

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

    root = os.path.dirname(os.path.abspath(__file__))
    model_dir = args.model_dir or detect_model_dir(root)
    if not model_dir:
        log("Model directory not found. Set --model_dir or WAN22_MODEL_DIR."); return 2

    width, height = normalize_resolution(model_dir, int(args.width), int(args.height))

    # Seed (optional)
    try:
        import torch
        g = torch.Generator(device="cuda")
        if args.seed: g.manual_seed(args.seed)
    except Exception:
        g = None

    # Build pipeline
    log(f"Loading model: {model_dir}")
    pipe = build_pipe(model_dir, args.dtype)

    # Move model weights to GPU
    try:
        pipe.to("cuda")
    except Exception as e:
        log(f"Failed to move pipe to CUDA: {e}"); return 3

    # Free VRAM: keep VAE on CPU and enable slicing
    try:
        pipe.vae.to("cpu")
        try: pipe.enable_vae_slicing()
        except Exception: pass
        try: pipe.enable_attention_slicing()
        except Exception: pass
        log("VAE moved to CPU (float32) with slicing; attention slicing enabled.")
        # Optional: xFormers if present
        try:
            pipe.enable_xformers_memory_efficient_attention()
            log("xFormers memory-efficient attention enabled.")
        except Exception as _e:
            log(f"xFormers not available or failed: {_e}")
    except Exception as _e:
        log(f"VAE/attention memory tweaks skipped: {_e}")

    # Scheduler
    apply_wan_scheduler_fix(pipe, args.sampler, width, height)

    # Frames rule
    steps  = int(args.steps)
    frames = round_frames(int(args.frames))
    if frames != args.frames:
        log("`num_frames - 1` has to be divisible by 4. Rounding to the nearest number.")

    # Progress callback
    steps_total = max(1, steps)
    def _cb(step, timestep, latents):
        cur = int(step) + 1
        pct = int(min(100, max(0, round(cur * 100 / steps_total))))
        print(f"[PROGRESS] step={cur}/{steps_total} frame=1/{frames} percent={pct}", flush=True)

    common = dict(
        prompt=args.prompt, negative_prompt=args.neg_prompt or None,
        width=width, height=height, num_inference_steps=steps,
        guidance_scale=args.cfg, num_frames=frames, generator=g, output_type="np",
    )

    # Call with/without callback depending on diffusers version
    result, last_err = None, None
    for with_cb in (True, False):
        try:
            kwargs = dict(common)
            if with_cb:
                kwargs["callback"] = _cb; kwargs["callback_steps"] = 1
            log(f"Generating… steps={steps} frames={frames} fps={args.fps}")
            result = pipe(**kwargs); break
        except TypeError as e:
            last_err = e; continue

    if result is None:
        log(f"Pipeline call failed: {last_err}"); return 4

    # Extract frames
    frames_out = None
    if isinstance(result, dict):
        frames_out = result.get("frames") or result.get("images")
    else:
        frames_out = getattr(result, "frames", None) or getattr(result, "images", None)
    if frames_out is None:
        log("No frames returned from pipeline"); return 5

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base  = f"WAN22_{args.mode}_{stamp}"
    json_path = os.path.join(args.outdir, base + ".json")
    mp4_path  = os.path.join(args.outdir, base + ".mp4")

    meta = {
        "mode": args.mode, "prompt": args.prompt, "neg_prompt": args.neg_prompt,
        "sampler": args.sampler, "steps": steps, "cfg": args.cfg, "seed": args.seed,
        "fps": args.fps, "frames": frames, "width": width, "height": height, "model_dir": model_dir,
    }
    with open(json_path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
    log(f"wrote: {json_path}")

    save_video(frames_out, args.fps, mp4_path)
    log(f"wrote: {mp4_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
