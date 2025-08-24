#!/usr/bin/env python
"""Gradio-based WAN 2.2 front-end for A1111 style workflow."""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from pathlib import Path
from typing import List

import gradio as gr

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "wan_runner.ps1"
DEFAULT_MODEL_DIR = "D:/wan22/models/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_OUTDIR = "D:/wan22/outputs"


def snap32(v: int) -> int:
    return max(32, v // 32 * 32)


def build_args(values: dict) -> List[str]:
    args = ["-ExecutionPolicy", "Bypass", "-File", str(RUNNER)]
    for key, val in values.items():
        if val is None or val == "":
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                args.append(flag)
        else:
            args.extend([flag, str(val)])
    return ["powershell"] + args


def run_cmd(**kw):
    mode = kw["mode"]
    prompt = kw["prompt"] or ""
    neg = kw["neg_prompt"] or ""
    image = kw.get("image")

    kw["width"] = snap32(int(kw["width"]))
    kw["height"] = snap32(int(kw["height"]))

    if kw["frames"] < 1 or kw["steps"] < 1 or kw["batch_count"] < 1 or kw["batch_size"] < 1:
        raise gr.Error("steps, frames, batch_count and batch_size must be >=1")
    if mode in {"t2v", "ti2v", "t2i"} and not prompt.strip():
        raise gr.Error("prompt required for text modes")
    if mode in {"i2v", "ti2v"} and not image:
        raise gr.Error("image required for image modes")

    if not Path(kw["model_dir"]).exists():
        raise gr.Error(f"model dir not found: {kw['model_dir']}")

    if mode == "t2i":
        kw["frames"] = 1
    args = {
        "mode": mode,
        "prompt": prompt,
        "neg_prompt": neg,
        "sampler": kw["sampler"],
        "steps": kw["steps"],
        "cfg": kw["cfg"],
        "seed": kw["seed"],
        "fps": kw["fps"],
        "frames": kw["frames"],
        "width": kw["width"],
        "height": kw["height"],
        "batch_count": kw["batch_count"],
        "batch_size": kw["batch_size"],
        "outdir": kw["outdir"],
        "model_dir": kw["model_dir"],
        "dtype": kw["dtype"],
        "attn": kw["attn"],
        "image": image,
    }
    cmd = build_args(args)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = deque(maxlen=200)
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        yield "\n".join(lines)
    proc.wait()
    lines.append(f"[exit {proc.returncode}]")
    yield "\n".join(lines)


def build_ui():
    with gr.Blocks(title="WAN 2.2 GUI") as demo:
        with gr.Column():
            mode = gr.Dropdown(["t2v", "i2v", "ti2v", "t2i"], value="t2v", label="Mode")
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
            dtype = gr.Dropdown(["bfloat16", "float16", "float32"], value="bfloat16", label="DType")
            attn = gr.Dropdown(["auto", "flash", "sdpa", "math"], value="auto", label="Attention")
            run = gr.Button("Generate")
            log = gr.Textbox(label="Log", lines=20)

        res_preset.change(
            lambda p: (
                (896, 512) if p == "896x512" else (960, 544) if p == "960x544" else (gr.update(), gr.update())
            ),
            inputs=res_preset,
            outputs=[width, height],
        )

        run.click(
            run_cmd,
            inputs=[
                mode,
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
