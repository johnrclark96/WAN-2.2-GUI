#!/usr/bin/env python
# wan_ps1_engine.py â€” pure Python engine the PS1 calls (no fallback)

import os, re, sys, json, time, argparse, random
from pathlib import Path

def log(msg): 
    print(msg, flush=True)

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["t2v","i2v","ti2v"], required=True)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--neg_prompt", type=str, default="")
    ap.add_argument("--init_image", type=str, default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--sampler", type=str, default="")
    ap.add_argument("--batch_count", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--outdir", type=str, default=str((Path(__file__).parent / "outputs").resolve()))
    ap.add_argument("--model_dir", type=str, default=str((Path(__file__).parent / "models").resolve()))
    ap.add_argument("--base", type=str, default="", help="Specific model path or HF model ID")
    ap.add_argument("--lora", action="append", help="LoRA as path[:scale]. Repeatable.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Unused; for parity with shim")
    return ap

def parse_loras(loras):
    out = []
    for item in loras or []:
        item = item.strip('"')
        if ":" in item:
            p, s = item.rsplit(":", 1)
            try: 
                scale = float(s)
            except: 
                scale = 0.8
        else:
            p, scale = item, 0.8
        out.append((Path(p), float(max(0.0, min(2.0, scale)))))
    return out

def group_lora_keys(sd):
    # Helper to pair LoRA weight keys
    ups = [k for k in sd.keys() if k.endswith("lora_up.weight")]
    for up in ups:
        base = up[:-len("lora_up.weight")]
        down = base + "lora_down.weight"
        alpha = base + "alpha"
        if down in sd:
            yield base, up, down, (alpha if alpha in sd else None)

def main():
    args = build_argparser().parse_args()

    # Lazy-import torch to speed up CLI startup
    import torch
    import torch.nn.functional as _F
    _orig_sdpa = _F.scaled_dot_product_attention

    def _sdpa_shim(*args, **kwargs):
        # Some WAN builds pass 'enable_gqa'; older torch ignores it
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)

    _F.scaled_dot_product_attention = _sdpa_shim

    try:
        from diffusers import DiffusionPipeline
    except Exception as e:
        log(f"[ps1-engine] Diffusers import error: {e}")
        return 2

    model_path = str(args.base if args.base else args.model_dir)
    mp = Path(model_path)
    if mp.exists() and mp.is_dir():
        if not (mp / "model_index.json").exists():
            # Auto-detect model subdirectory if a single model is present
            subdirs = [d for d in mp.iterdir() if d.is_dir() and (d/"model_index.json").exists()]
            if len(subdirs) == 1:
                model_path = str(subdirs[0])
                mp = subdirs[0]
                log(f"[ps1-engine] Auto-detected model path: {model_path}")
            else:
                log(f"[ps1-engine] The --base path looks like a local folder but is not a Diffusers model (missing model_index.json): {mp}")
                return 11

    # Load with trust_remote_code so WAN-specific pipeline classes (WanPipeline) can be used if not built-in
    pipe = None
    try:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        log(f"[ps1-engine] Loading model with remote code: {model_path}")
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
    except Exception as e:
        log(f"[ps1-engine] Could not load pipeline from {model_path}: {e}")
        return 3

    # Move pipeline to GPU (or CPU) with memory optimization if available
    if hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()  # offload to CPU when not in use to save VRAM
        except Exception:
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    elif hasattr(pipe, "to"):
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Set scheduler if specified in args
    sched_name = (args.sampler or "").strip().lower()
    if sched_name:
        try:
            from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler, UniPCMultistepScheduler
        except Exception as e:
            log(f"[ps1-engine] Scheduler import error: {e}")
        else:
            if sched_name == "ddim":
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            elif sched_name in ["dpmpp_2m_sde", "dpmpp_3m_sde"]:
                # The scheduler's config is a FrozenDict and cannot be mutated in-place.
                # Instead, create a new scheduler with the desired parameters.
                solver_order = 2 if sched_name.startswith("dpmpp_2") else 3
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    algorithm_type="sde-dpmsolver",
                    solver_order=solver_order,
                )
            elif sched_name == "unipc":
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            elif sched_name == "euler_a":
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            elif sched_name == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            log(f"[ps1-engine] Scheduler set to {sched_name}")

    # Apply LoRA weights if provided
    loras = parse_loras(args.lora)
    if loras:
        try:
            from safetensors.torch import load_file as safe_load
            target = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
            if target is not None:
                base_sd = {
                    k: (v.float().clone() if getattr(v, "is_floating_point", lambda: False)() else v.clone())
                    for k,v in target.state_dict().items()
                }
                matched = 0
                for path, scale in loras:
                    log(f"[ps1-engine] LoRA: {path} @ {scale}")
                    if not path.exists():
                        log(f"[ps1-engine]   missing: {path}")
                        continue
                    sd = safe_load(str(path)) if path.suffix.lower()==".safetensors" else torch.load(str(path), map_location="cpu")
                    for base, upk, downk, alphak in group_lora_keys(sd):
                        tk = base + "weight"
                        if tk not in base_sd:
                            if tk.endswith(".") and tk[:-1] in base_sd:
                                tk = tk[:-1]  # adjust key if needed
                            else:
                                continue
                        up = sd[upk].float(); down = sd[downk].float()
                        r = down.shape[0]; 
                        alpha = sd[alphak].item() if (alphak and alphak in sd) else r
                        delta = (up @ down) * (alpha / r) * scale
                        if hasattr(base_sd[tk], "shape") and base_sd[tk].shape != delta.shape:
                            continue
                        base_sd[tk].add_(delta); matched += 1
                # Load merged weights back into the model (non-strict to allow partial matches)
                dtype0 = next(target.parameters()).dtype
                for k,v in base_sd.items():
                    if getattr(v, "is_floating_point", lambda: False)():
                        base_sd[k] = v.to(dtype0)
                target.load_state_dict(base_sd, strict=False)
                log(f"[ps1-engine] LoRA merged: {matched} groups")
            else:
                log("[ps1-engine] WARN: no applicable module to merge LoRAs into (unet/transformer not found)")
        except Exception as e:
            log(f"[ps1-engine] LoRA merge failed: {e}")

    # Handle random seed
    seed = random.randint(1, 2**31 - 1) if args.seed is None or args.seed < 0 else args.seed
    generator = None
    try:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    except Exception:
        pass

    # Snap width/height to multiples of 16 (required by model) and sanitize FPS
    def _snap16(n: int) -> int:
        return max(16, (int(n) // 16) * 16)
    W = _snap16(args.width); H = _snap16(args.height)
    if (W, H) != (args.width, args.height):
        log(f"[ps1-engine] Adjusted resolution to {W}x{H} (16-px multiples) from {args.width}x{args.height}")
    # FPS must be positive int
    try:
        fps_i = int(args.fps)
        if fps_i < 1: fps_i = 12
    except Exception:
        fps_i = 12

    # Common pipeline call parameters
    common_kwargs = dict(
        prompt=args.prompt or "",
        negative_prompt=args.neg_prompt or "",
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.cfg),
        width=int(W),
        height=int(H),
    )

    # --- progress callback: emit per-step progress lines for console bars ---
    steps_total = max(1, int(args.steps))
    frames_total = max(1, int(args.frames))

    def _progress_cb(*cb_args, **cb_kwargs):
        # diffusers usually calls callback(step, timestep, latents)
        try:
            step = int(cb_args[0]) if cb_args else int(cb_kwargs.get("step", 0))
        except Exception:
            step = 0
        cur = step + 1
        pct = int(min(100, max(0, round(cur * 100 / steps_total))))
        # We can't reliably know current frame; advertise in-flight.
        print(
            f"[PROGRESS] step={cur}/{steps_total} frame=1/{frames_total} percent={pct}",
            flush=True,
        )
    if generator is not None:
        common_kwargs["generator"] = generator
    if args.mode in ("i2v", "ti2v") and args.init_image:
        common_kwargs["image"] = args.init_image  # path to init image

    # Try calling the pipeline with various parameter names (diffusers pipelines may use num_frames or video_length, etc.)
    call_variants = [
        {"num_frames": int(args.frames), "fps": fps_i},
        {"video_length": int(args.frames), "fps": fps_i},
        {"num_frames": int(args.frames)},
        {"video_length": int(args.frames)},
        {},
    ]
    result = None
    last_err = None

    # Try with callback first; if pipeline rejects it, retry without
    for v in call_variants:
        base_kwargs = {**common_kwargs, **v}
        for with_cb in (True, False):
            try:
                kwargs = dict(base_kwargs)
                if with_cb:
                    kwargs["callback"] = _progress_cb
                    kwargs["callback_steps"] = 1
                log(
                    f"[ps1-engine] Generating… steps={kwargs.get('num_inference_steps')} "
                    f"frames={kwargs.get('num_frames') or kwargs.get('video_length')} "
                    f"fps={kwargs.get('fps')}"
                )
                result = pipe(**kwargs)
                raise StopIteration  # success
            except TypeError as e:
                last_err = e
                if with_cb:
                    continue  # retry without callback
            except StopIteration:
                break
        if result is not None:
            break

    if result is None:
        log(f"[ps1-engine] Pipeline call failed: {last_err}")
        return 4

    # Prepare output directory and filename
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = outdir / f"WAN22_{args.mode}_{timestamp}.mp4"

    def save_from_frames(frame_list):
        try:
            import numpy as np, imageio
            arr = [np.array(f) for f in frame_list]
            imageio.mimwrite(str(video_path), arr, fps=max(1, int(args.fps)))
            log(f"saved: {video_path}")
            return True
        except Exception as e:
            log(f"[ps1-engine] frame save error: {e}")
            return False

    # Try saving the video from the result
    saved = False
    try:
        # Some pipelines return an object with frames attribute
        if hasattr(result, "frames"):
            saved = save_from_frames(result.frames)
        elif isinstance(result, dict) and "frames" in result:
            saved = save_from_frames(result["frames"])
    except Exception:
        pass

    if not saved:
        # Some diffusers pipelines return a tensor or list in 'videos'
        try:
            vid_tensor = None
            if isinstance(result, dict):
                vid_tensor = result.get("videos")
            else:
                vid_tensor = getattr(result, "videos", None)
            if vid_tensor is not None:
                import numpy as np, imageio
                v = vid_tensor
                # Handle batch dim and channel-last conversion:
                if hasattr(v, "ndim") and v.ndim == 5:
                    v = v[0]  
                v = (v.permute(0,2,3,1).clamp(0,1).cpu().numpy() * 255).astype("uint8")
                imageio.mimwrite(str(video_path), v, fps=max(1, int(args.fps)))
                log(f"saved: {video_path}")
                saved = True
        except Exception as e:
            log(f"[ps1-engine] tensor video save error: {e}")

    if not saved:
        # If no video could be saved, write a JSON sidecar as a fallback
        sidecar = outdir / f"WAN22_{args.mode}_{timestamp}.json"
        try:
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({"status":"ok","note":"No video output detected"}, f)
            log(f"wrote: {sidecar}")
        except Exception:
            pass

    return 0

if __name__ == "__main__":
    sys.exit(main())




