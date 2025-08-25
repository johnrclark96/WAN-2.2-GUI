#!/usr/bin/env python
"""Gradio-based WAN 2.2 front-end for A1111 style workflow."""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
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


def run_cmd(engine: str, **kw):
    mode = kw["mode"]
    prompt = kw["prompt"] or ""
    neg = kw["neg_prompt"] or ""
    image = kw.get("image")

    kw["width"] = snap32(int(kw["width"]))
    kw["height"] = snap32(int(kw["height"]))

    if kw["frames"] < 1 or kw["steps"] < 1 or kw["batch_count"] < 1 or kw["batch_size"] < 1:
        raise gr.Error("steps, frames, batch_count and batch_size must be >=1")
    if mode in {"t2v", "ti2v"} and not prompt.strip():
        raise gr.Error("prompt required for text modes")
    if mode in {"i2v", "ti2v"} and not image:
        raise gr.Error("image required for image modes")

    if engine == "official":
        if not paths.OFFICIAL_GENERATE or not Path(paths.OFFICIAL_GENERATE).exists():
            raise gr.Error("Set OFFICIAL_GENERATE path in Paths tab.")
        w = kw["width"]
        h = kw["height"]
        w = snap32(int(w))
        h = snap32(int(h))
        if h != 704:
            raise gr.Error("Official engine only supports height=704 (720p).")
        size = f"{w}*{h}"
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
            size,
        ]
        if image:
            cmd.extend(["--image", str(Path(image).resolve())])
        try:
            import torch
            free, _ = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            if free / 1024**3 < 24:
                cmd.extend(["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"])
        except Exception:
            pass
    else:
        if not Path(kw["model_dir"]).exists():
            raise gr.Error(f"model dir not found: {kw['model_dir']}")
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
            engine = gr.Radio(["diffusers", "official"], value="diffusers", label="Engine")
            mode = gr.Dropdown(["t2v", "i2v", "ti2v"], value="t2v", label="Mode")
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

        def on_engine_change(e):
            disabled = e == "official"
            upd = gr.update(interactive=not disabled)
            return [upd, upd, upd, upd, upd, upd]

        engine.change(
            on_engine_change,
            inputs=engine,
            outputs=[sampler, steps, cfg, fps, frames, attn],
        )

        run.click(
            lambda eng, *vals: run_cmd(eng, **{
                "mode": vals[0],
                "prompt": vals[1],
                "neg_prompt": vals[2],
                "sampler": vals[3],
                "steps": vals[4],
                "cfg": vals[5],
                "seed": vals[6],
                "fps": vals[7],
                "frames": vals[8],
                "width": vals[9],
                "height": vals[10],
                "batch_count": vals[11],
                "batch_size": vals[12],
                "outdir": vals[13],
                "model_dir": vals[14],
                "dtype": vals[15],
                "attn": vals[16],
                "image": vals[17],
            }),
            inputs=[engine, mode, prompt, neg_prompt, sampler, steps, cfg, seed, fps, frames, width, height, batch_count, batch_size, outdir, model_dir, dtype, attn, image],
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
