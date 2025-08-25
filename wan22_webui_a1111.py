#!/usr/bin/env python
"""Gradio-based WAN 2.2 front-end for A1111 style workflow.

This file defines a small Gradio app and a streaming runner that launches the
PowerShell shim (wan_runner.ps1). It also supports an "official" engine path
that calls the reference generate.py via the configured virtual environment
interpreter.
"""

from __future__ import annotations

import argparse
import json
import select
import subprocess
from pathlib import Path
from typing import Any, Generator, List


from core import paths

import gradio as gr


APP_TITLE = "WAN 2.2 GUI"

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "wan_runner.ps1"

DEFAULT_MODEL_DIR = (paths.MODELS_DIR / "Wan2.2-TI2V-5B-Diffusers").as_posix()
DEFAULT_OUTDIR = paths.OUTPUT_DIR.as_posix()


def snap32(v: int) -> int:
    return max(32, (int(v) // 32) * 32)


def safe_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        val = int(value)
    except Exception:
        val = default
    if minimum is not None:
        val = max(minimum, val)
    return val


def safe_float(value: Any, default: float, minimum: float | None = None) -> float:
    try:
        val = float(value)
    except Exception:
        val = default
    if minimum is not None:
        val = max(minimum, val)
    return val


def build_args(values: dict) -> List[str]:
    """Build the PowerShell invocation to the engine shim with CLI args."""
    args: List[str] = ["pwsh", "-NoLogo", "-File", RUNNER.as_posix()]
    order = [
        "mode", "prompt", "neg_prompt", "sampler", "steps", "cfg", "seed",
        "fps", "frames", "width", "height", "batch_count", "batch_size",
        "outdir", "model_dir", "dtype", "attn", "image",
    ]
    for key in order:
        val = values.get(key)
        if val is None or val == "" or (isinstance(val, bool) and not val):
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            args.append(flag)
        else:
            args.extend([flag, str(val)])
    return args


def estimate_mem(width: int, height: int, frames: int, micro: int, batch: int) -> float:
    return width * height * frames * micro * batch / 1024 ** 3 * 2e-6


def stream_run(cmd: List[str]) -> Generator[str, None, None]:
    """Launch *cmd* and yield combined stdout/stderr lines."""
    yield "[launch] " + " ".join(cmd)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        yield f"[error] {e}"
        return

    assert proc.stdout is not None and proc.stderr is not None
    streams = {proc.stdout: "", proc.stderr: "[stderr] "}
    fds = list(streams.keys())
    while fds:
        ready, _, _ = select.select(fds, [], [])
        for stream in ready:
            line = stream.readline()
            if not line:
                fds.remove(stream)
                continue
            stripped = line.rstrip()
            try:
                json.loads(stripped)
            except Exception:
                pass
            yield streams[stream] + stripped
    code = proc.wait()
    yield f"[exit] code={code}"


def run_cmd(engine: str, **kw) -> Generator[str, None, None]:
    mode = kw["mode"]
    prompt = kw.get("prompt", "")
    image = kw.get("image")

    # unified input validation
    prompt_str = (prompt or "").strip()
    image_path = image or ""

    # general sanity checks
    if not prompt_str and not image_path:
        raise gr.Error("Either a prompt or an image is required.")
    if image_path and not Path(image_path).exists():
        raise gr.Error(f"image not found: {image_path}")

    # mode-specific guarantees
    if mode in {"t2v", "ti2v", "t2i"} and not prompt_str:
        raise gr.Error("prompt required for text modes")
    if mode in {"i2v", "ti2v"} and not image_path:
        raise gr.Error("image required for image modes")

    # frame and size normalization
    req_w = snap32(int(kw["width"]))
    req_h = snap32(int(kw["height"]))
    req_frames = int(kw["frames"])
    if (req_frames - 1) % 4 != 0:
        req_frames = (req_frames - 1) // 4 * 4 + 1
    w, h, frames = req_w, req_h, req_frames

    # update for downstream display in logs
    kw.update({"width": w, "height": h, "frames": frames})

    if engine == "official":
        if not paths.OFFICIAL_GENERATE or not Path(paths.OFFICIAL_GENERATE).exists():
            raise gr.Error("Set OFFICIAL_GENERATE path in the Paths tab.")
        if h != 704:
            raise gr.Error("Official engine only supports height=704 (720p).")
        size = f"{w}*{h}"
        cmd: List[str] = [
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
        if image_path:
            cmd += ["--image", str(Path(image_path).resolve())]
        # memory guard
        try:
            import torch  # type: ignore
            free, _ = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            if free / 1024 ** 3 < 24:
                cmd += ["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"]
        except Exception:
            pass
    else:
        # diffusers engine via PowerShell runner
        args = dict(kw)
        args.update({"mode": mode})
        cmd = build_args(args)

    yield from stream_run(cmd)


def build_ui():
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"## {APP_TITLE}", elem_id="app-title")

        engine = gr.Radio(["diffusers", "official"], value="diffusers", label="Engine")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=3)
            neg_prompt = gr.Textbox(label="Negative Prompt", lines=2)

        with gr.Row():
            sampler = gr.Textbox(value="euler", label="Sampler")
            steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
            cfg = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG")
            seed = gr.Number(value=-1, label="Seed")

        with gr.Row():
            fps = gr.Slider(1, 30, value=24, step=1, label="FPS")
            frames = gr.Slider(1, 49, value=9, step=1, label="Frames")
            width = gr.Number(value=1280, label="Width")
            height = gr.Number(value=704, label="Height")

        with gr.Row():
            batch_count = gr.Number(value=1, label="Batch Count")
            batch_size = gr.Number(value=1, label="Batch Size")
            dtype = gr.Dropdown(["fp16", "bf16", "fp32"], value="bf16", label="DType")
            attn = gr.Dropdown(["sdpa", "flash-attn"], value="sdpa", label="Attention")

        with gr.Row():
            model_dir = gr.Textbox(value=DEFAULT_MODEL_DIR, label="Model Dir")
            outdir = gr.Textbox(value=DEFAULT_OUTDIR, label="Output Dir")

        image = gr.Image(label="Init Image", type="filepath")

        run = gr.Button("Generate")
        log = gr.Textbox(label="Log", lines=15)

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

            # Official engine does not support pure image-to-video in this GUI
            if eng == "official" and mode_v == "i2v":
                raise gr.Error("Use the diffusers engine for image-to-video.")

            gr.Info(f"mode={mode_v}")

            # Sanitize numeric inputs
            steps_v = safe_int(steps_v, 20, 1)
            cfg_v = safe_float(cfg_v, 7.0, 0.0)
            fps_v = safe_int(fps_v, 24, 1)
            frames_v = safe_int(frames_v, 9, 1)
            if (frames_v - 1) % 4 != 0:
                frames_v = (frames_v - 1) // 4 * 4 + 1
            width_v = snap32(safe_int(width_v, 1280, 32))
            height_v = snap32(safe_int(height_v, 704, 32))
            batch_count_v = safe_int(batch_count_v, 1, 1)
            batch_size_v = safe_int(batch_size_v, 1, 1)
            try:
                seed_v = int(seed_v)
            except Exception:
                seed_v = -1

            # Command-line-safe mode sent to the runner
            if eng == "diffusers":
                cli_mode = "t2i" if int(frames_v) == 1 else "t2v"
            else:  # eng == "official"
                cli_mode = mode_v

            for line in run_cmd(
                eng,
                mode=cli_mode,
                prompt=str(prompt_v or ""),
                neg_prompt=str(neg_v or ""),
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
    ap.add_argument("--listen", action="store_true", default=False)
    ap.add_argument("--auth", type=str, default="")
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
