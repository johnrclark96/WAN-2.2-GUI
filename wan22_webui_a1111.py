#!/usr/bin/env python
"""Gradio-based WAN 2.2 front-end for A1111 style workflow."""

from __future__ import annotations

import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List

from core import paths

import gradio as gr


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "wan_runner.ps1"
DEFAULT_MODEL_DIR = (paths.MODELS_DIR / "Wan2.2-TI2V-5B-Diffusers").as_posix()
DEFAULT_OUTDIR = paths.OUTPUT_DIR.as_posix()


def snap32(v: int) -> int:
    return max(32, v // 32 * 32)


def build_args(values: dict) -> List[str]:
    args = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(RUNNER),
    ]
    for key, val in values.items():
        if val is None or val == "":
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                args.append(flag)
        else:
            args.extend([flag, str(val)])
    return args


def estimate_mem(width: int, height: int, frames: int, micro: int, batch: int) -> float:
    return width * height * frames * micro * batch / 1024 ** 3 * 2e-6


def run_cmd(engine: str, **kw):
    mode = kw["mode"]
    prompt = kw.get("prompt", "")
    neg = kw.get("neg_prompt", "")
    image = kw.get("image")

    req_w = snap32(int(kw["width"]))
    req_h = snap32(int(kw["height"]))
    req_frames = int(kw["frames"])
    if (req_frames - 1) % 4 != 0:
        req_frames = (req_frames - 1) // 4 * 4 + 1
    w, h, frames = req_w, req_h, req_frames
    kw.update({"width": w, "height": h, "frames": frames})

    if frames < 1 or kw["steps"] < 1 or kw["batch_count"] < 1 or kw["batch_size"] < 1:
        raise gr.Error("steps, frames, batch_count and batch_size must be >=1")
    if not prompt.strip() and not image:
        raise gr.Error("prompt required when no image provided")
    if image and not Path(image).exists():
        raise gr.Error(f"image not found: {image}")

    if engine == "official":
        if mode == "i2v":
            raise gr.Error("Use the diffusers engine for image-to-video.")
        if not paths.OFFICIAL_GENERATE or not Path(paths.OFFICIAL_GENERATE).exists():
            raise gr.Error("Set OFFICIAL_GENERATE path in Paths tab.")
        if h != 704:
            raise gr.Error("Official engine requires height=704. Try 1280x704 or 704x1280.")
        cmd = [
            paths.VENV_PY.as_posix(),
            paths.OFFICIAL_GENERATE,
            "--task",
            "ti2v-5B",
            "--prompt",
            prompt,
            "--ckpt_dir",
            paths.CKPT_TI2V_5B.as_posix(),
            "--size",
            f"{w}*{h}",
        ]
        if mode == "ti2v" and image:
            cmd += ["--image", str(Path(image).resolve())]
        try:
            import torch
            free, _ = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            if free / 1024 ** 3 < 24:
                cmd += ["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"]
        except Exception:
            pass
        effective_line = None
    else:
        if not Path(kw["model_dir"]).exists():
            raise gr.Error(f"model dir not found: {kw['model_dir']}")
        attn_mode = kw["attn"]
        if attn_mode == "flash-attn":
            try:
                import torch
                major, _ = torch.cuda.get_device_capability()  # type: ignore[attr-defined]
                if major < 9:
                    gr.Warning("FlashAttention v3 requires Hopper; using sdpa")
                    attn_mode = "sdpa"
            except Exception:
                gr.Warning("FlashAttention v3 requires Hopper; using sdpa")
                attn_mode = "sdpa"

        micro = 128
        batch_size = int(kw["batch_size"])
        free_gb = total_gb = 0.0
        try:
            import torch
            free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            free_gb, total_gb = free / 1024 ** 3, total / 1024 ** 3
        except Exception:
            free_gb = 32.0
            total_gb = free_gb
        req = estimate_mem(w, h, frames, micro, batch_size)
        while req > free_gb and micro > 32:
            micro //= 2
            req = estimate_mem(w, h, frames, micro, batch_size)
        while req > free_gb and frames > 1:
            frames = max(1, frames - 4)
            if (frames - 1) % 4 != 0:
                frames = (frames - 1) // 4 * 4 + 1
            req = estimate_mem(w, h, frames, micro, batch_size)
        aspect = w / h
        while req > free_gb and w > 32 and h > 32:
            w = snap32(int(w * 0.9))
            h = snap32(int(w / aspect))
            req = estimate_mem(w, h, frames, micro, batch_size)
        kw.update({"width": w, "height": h, "frames": frames})
        effective = {
            "width": w,
            "height": h,
            "frames": frames,
            "micro_batch": micro,
            "batch_size": batch_size,
        }
        paths.JSON_DIR.mkdir(parents=True, exist_ok=True)
        sidecar = paths.JSON_DIR / f"WAN22_{datetime.now():%Y%m%d_%H%M%S}.json"
        requested = {
            "width": req_w,
            "height": req_h,
            "frames": req_frames,
            "micro_batch": 128,
            "batch_size": batch_size,
        }
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "requested": requested,
                    "effective": effective,
                    "memory": {"free_gb": free_gb, "total_gb": total_gb},
                },
                fh,
                indent=2,
            )
        effective_line = (
            f"effective settings: width={w} height={h} frames={frames} micro_batch={micro} batch_size={batch_size}"
        )
        args = {
            "mode": mode,
            "prompt": prompt,
            "neg_prompt": neg,
            "sampler": kw["sampler"],
            "steps": kw["steps"],
            "cfg": kw["cfg"],
            "seed": kw["seed"],
            "fps": kw["fps"],
            "frames": frames,
            "width": w,
            "height": h,
            "batch_count": kw["batch_count"],
            "batch_size": batch_size,
            "outdir": kw["outdir"],
            "model_dir": kw["model_dir"],
            "dtype": kw["dtype"],
            "attn": attn_mode,
            "image": image,
        }
        cmd = build_args(args)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    yield "launch: " + " ".join(cmd)
    if effective_line:
        yield effective_line
    for line in proc.stdout:
        yield line.rstrip()
    proc.wait()
    yield f"[exit {proc.returncode}]"


def build_ui():
    with gr.Blocks(title="WAN 2.2 GUI") as demo:
        with gr.Tabs():
            with gr.Tab("Generate"):
                engine = gr.Radio(["diffusers", "official"], value="diffusers", label="Engine")
                prompt = gr.Textbox(label="Prompt", lines=3)
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                image = gr.Image(type="filepath", label="Init Image")
                sampler = gr.Dropdown(["unipc"], value="unipc", label="Sampler")
                steps = gr.Slider(1, 100, value=20, step=1, label="Steps")
                cfg = gr.Slider(0, 20, value=7.0, step=0.5, label="CFG")
                seed = gr.Number(value=-1, label="Seed (-1=random)")
                fps = gr.Slider(1, 60, value=24, step=1, label="FPS")
                frames = gr.Slider(1, 200, value=16, step=1, label="Frames")
                width = gr.Number(value=768, label="Width")
                height = gr.Number(value=432, label="Height")
                res_preset = gr.Dropdown(
                    ["(none)", "896x512", "960x544"],
                    value="(none)",
                    label="Resolution Preset (16GB friendly)",
                )
                batch_count = gr.Number(value=1, label="Batch Count")
                batch_size = gr.Number(value=1, label="Batch Size")
                model_dir = gr.Textbox(value=DEFAULT_MODEL_DIR, label="Model Dir")
                outdir = gr.Textbox(value=DEFAULT_OUTDIR, label="Output Dir")
                dtype = gr.Dropdown(["bf16", "fp16", "fp32"], value="bf16", label="DType")
                attn = gr.Dropdown(["sdpa", "flash-attn"], value="sdpa", label="Attention")
                run = gr.Button("Generate")
                log = gr.Textbox(label="Log", lines=20)

            with gr.Tab("Paths"):
                official_path = gr.Textbox(label="Official generate.py", value=paths.OFFICIAL_GENERATE)
                save_path = gr.Button("Save")

                def _save(p):
                    paths.save_config({"OFFICIAL_GENERATE": p})
                    return p

                save_path.click(_save, inputs=official_path, outputs=official_path)

        res_preset.change(
            lambda p: (
                (896, 512) if p == "896x512" else (960, 544) if p == "960x544" else (gr.update(), gr.update())
            ),
            inputs=res_preset,
            outputs=[width, height],
        )

        def on_engine_change(e):
            if e == "official" and (
                not paths.OFFICIAL_GENERATE or not Path(paths.OFFICIAL_GENERATE).exists()
            ):
                raise gr.Error("Select official generate.py in Paths tab first.")
            disabled = e == "official"
            upd = gr.update(interactive=not disabled)
            return [upd, upd, upd, upd, upd, upd, upd, upd]

        engine.change(
            on_engine_change,
            inputs=engine,
            outputs=[sampler, steps, cfg, fps, frames, attn, neg_prompt, batch_size],
        )

        def on_run(
            eng,
            prompt_v,
            neg_v,
            sampler_v,
            steps_v,
            cfg_v,
            seed_v,
            fps_v,
            frames_v,
            width_v,
            height_v,
            batch_count_v,
            batch_size_v,
            outdir_v,
            model_dir_v,
            dtype_v,
            attn_v,
            image_v,
        ):
            # Human-readable mode for user feedback
            mode_v = "ti2v" if (prompt_v and image_v) else ("i2v" if image_v else "t2v")

            # Official engine does not support pure image-to-video in this graphical user interface
            if eng == "official" and mode_v == "i2v":
                raise gr.Error("Use the diffusers engine for image-to-video.")

            gr.Info(f"mode={mode_v}")

            # Command line safe mode that is actually sent to the runner
            if eng == "diffusers":
                cli_mode = "t2i" if int(frames_v) == 1 else "t2v"
            else:  # eng == "official"
                cli_mode = mode_v

            for line in run_cmd(
                eng,
                mode=cli_mode,
                prompt=prompt_v,
                neg_prompt=neg_v,
                sampler=sampler_v,
                steps=steps_v,
                cfg=cfg_v,
                seed=seed_v,
                fps=fps_v,
                frames=frames_v,
                width=width_v,
                height=height_v,
                batch_count=batch_count_v,
                batch_size=batch_size_v,
                outdir=outdir_v,
                model_dir=model_dir_v,
                dtype=dtype_v,
                attn=attn_v,
                image=image_v,
            ):
                yield line

        # Single click binding that streams from on_run
        run.click(
            on_run,
            inputs=[
                engine,
                prompt,
                neg_prompt,
                sampler,
                steps,
                cfg,
                seed,
                fps,
                frames,
                width,
                height,
                batch_count,
                batch_size,
                outdir,
                model_dir,
                dtype,
                attn,
                image,
            ],
            outputs=log,
        )
    return demo


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--auth", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    ui = build_ui()
    auth = tuple(args.auth.split(":", 1)) if args.auth and ":" in args.auth else None
    ui.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        auth=auth,
        inbrowser=False,
        show_api=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
