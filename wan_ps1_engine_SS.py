#!/usr/bin/env python
# wan_ps1_engine.py — pure Python engine the PS1 calls (no fallback)

import os, re, sys, json, time, argparse, random, gc, contextlib
from pathlib import Path

def log(msg): print(msg, flush=True)

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
    ap.add_argument("--base", type=str, default="", help="Specific model path or HF id")
    ap.add_argument("--lora", action="append", help="LoRA as path[:scale]. Repeatable.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Unused; for parity with shim")
    ap.add_argument("--attn", choices=["auto", "flash", "sdpa"], default="auto")
    return ap

def parse_loras(loras):
    out = []
    for item in loras or []:
        item = item.strip('"')
        if ":" in item:
            p, s = item.rsplit(":", 1)
            try: scale = float(s)
            except: scale = 0.8
        else:
            p, scale = item, 0.8
        out.append((Path(p), float(max(0.0, min(2.0, scale)))))
    return out

def group_lora_keys(sd):
    ups = [k for k in sd.keys() if k.endswith("lora_up.weight")]
    for up in ups:
        base = up[:-len("lora_up.weight")]
        down = base + "lora_down.weight"
        alpha = base + "alpha"
        if down in sd:
            yield base, up, down, (alpha if alpha in sd else None)


def _detect_flash_attn() -> bool:
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

def main():
    args = build_argparser().parse_args()

    # lazy imports so PS starts fast
    import torch
    try:
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
    try:
        from diffusers import DiffusionPipeline
    except Exception as e:
        log(f"[ps1-engine] Diffusers import error: {e}")
        return 2

    model_path = str(args.base if args.base else args.model_dir)

    # If user points to a *local* folder, fail fast if it's not a proper Diffusers export
    mp = Path(model_path)
    if mp.exists() and mp.is_dir():
        if not (mp / "model_index.json").exists():
            log(f"[ps1-engine] The --base path looks like a local folder but is not a Diffusers model "
                f"(missing model_index.json): {mp}")
            return 11

    # Load with remote code so WAN-specific pipeline classes (WanPipeline) work
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
        log(f"[ps1-engine] Could not load pipeline with remote code: {e}")
        return 3

    # Device / offload
    offload_done = False
    if hasattr(pipe, "enable_sequential_cpu_offload"):
        try:
            pipe.enable_sequential_cpu_offload()
            log("[ps1-engine] Using sequential CPU offload")
            offload_done = True
        except Exception as e:
            log(f"[ps1-engine] Sequential CPU offload unavailable: {e}")
    if not offload_done and hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
            log("[ps1-engine] Using model CPU offload")
            offload_done = True
        except Exception as e:
            log(f"[ps1-engine] Model CPU offload failed: {e}")
    if not offload_done and hasattr(pipe, "to"):
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # LoRA merge (best-effort)
    loras = parse_loras(args.lora)
    if loras:
        try:
            from safetensors.torch import load_file as safe_load
            target = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
            if target is not None:
                base_sd = {k: (v.float().clone() if getattr(v, "is_floating_point", lambda: False)() and v.is_floating_point() else v.clone())
                           for k, v in target.state_dict().items()}
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
                                tk = tk[:-1]
                            else:
                                continue
                        up = sd[upk].float(); down = sd[downk].float()
                        r = down.shape[0]; alpha = sd[alphak].item() if (alphak and alphak in sd) else r
                        delta = (up @ down) * (alpha / r) * scale
                        if hasattr(base_sd[tk], "shape") and base_sd[tk].shape != delta.shape:
                            continue
                        base_sd[tk].add_(delta); matched += 1
                mdtype = next(target.parameters()).dtype
                for k, v in list(base_sd.items()):
                    if getattr(v, "is_floating_point", lambda: False)() and v.is_floating_point():
                        base_sd[k] = v.to(mdtype)
                target.load_state_dict(base_sd, strict=False)
                log(f"[ps1-engine] LoRA merged: {matched} groups")
            else:
                log("[ps1-engine] WARN: no unet/transformer to merge into")
        except Exception as e:
            log(f"[ps1-engine] LoRA merge failed: {e}")

    # Seed
    seed = random.randint(1, 2**31 - 1) if args.seed is None or args.seed < 0 else args.seed
    g = None
    try:
        g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    except Exception:
        pass

    # --- snap width/height to multiples of 16 and sanitize fps ---
    def _snap16(n: int) -> int:
        n = int(n)
        return max(16, (n // 16) * 16)

    W = _snap16(args.width)
    H = _snap16(args.height)
    if (W, H) != (args.width, args.height):
        log(f"[ps1-engine] Adjusted resolution to multiples of 16: {args.width}x{args.height} -> {W}x{H}")

    # fps must be a positive int; if invalid, default to 12
    try:
        fps_i = int(args.fps)
        if fps_i < 1:
            fps_i = 12
    except Exception:
        fps_i = 12

    common = dict(
        prompt=args.prompt or "",
        negative_prompt=args.neg_prompt or "",
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.cfg),
        width=int(W),
        height=int(H),
    )
    if g is not None:
        common["generator"] = g
    if args.mode in ("i2v", "ti2v") and args.init_image:
        common["image"] = args.init_image

    backend, attn_ctx = _select_attn_ctx(args.attn)
    if args.attn == "flash" and backend != "flash":
        log("[ps1-engine] flash_attn requested but unavailable; using SDPA")
    log(f"[ps1-engine] Attention backend: {backend}")

    # Try several call signatures; start with fps, then fall back gracefully
    variants = [
        {"num_frames": int(args.frames), "fps": fps_i},
        {"video_length": int(args.frames), "fps": fps_i},
        {"num_frames": int(args.frames)},
        {"video_length": int(args.frames)},
        {}
    ]

    out = None
    last_err = None
    for vkw in variants:
        try:
            kw = dict(common)
            kw.update(vkw)
            log(
                f"[ps1-engine] Generating... steps={kw.get('num_inference_steps')} "
                f"frames={kw.get('num_frames') or kw.get('video_length')} fps={kw.get('fps')}"
            )
            ctx = attn_ctx if attn_ctx is not None else contextlib.nullcontext()
            with torch.inference_mode(), ctx:
                out = pipe(**kw)
            break
        except TypeError as e:
            last_err = e
            continue

    if out is None:
        log(f"[ps1-engine] pipeline call failed: {last_err}")
        return 4

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    video_path = outdir / f"WAN22_{args.mode}_{ts}.mp4"

    def save_from_frames(frames):
        try:
            import numpy as np, imageio
            arr = [np.array(f) for f in frames]
            imageio.mimwrite(str(video_path), arr, fps=fps_i)
            log(f"saved: {video_path}")
            return True
        except Exception as e:
            log(f"[ps1-engine] frame save error: {e}"); return False

    saved = False
    try:
        if hasattr(out, "frames"):
            saved = save_from_frames(out.frames)
        elif isinstance(out, dict) and "frames" in out:
            saved = save_from_frames(out["frames"])
    except Exception:
        pass

    if not saved:
        try:
            vid = getattr(out, "videos", None) if not isinstance(out, dict) else out.get("videos")
            if vid is not None:
                import numpy as np, imageio
                v = vid[0] if getattr(vid, "ndim", 0) == 5 else vid
                v = (v.permute(0,2,3,1).clamp(0,1).cpu().numpy() * 255).astype("uint8")
                imageio.mimwrite(str(video_path), v, fps=fps_i)
                log(f"saved: {video_path}")
                saved = True
        except Exception as e:
            log(f"[ps1-engine] tensor video save error: {e}")

    if not saved:
        sidecar = outdir / f"WAN22_{args.mode}_{ts}.json"
        try:
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({"status":"ok","note":"No video detected"}, f)
            log(f"wrote: {sidecar}")
        except Exception:
            pass

    del out, pipe
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    sys.exit(main())
