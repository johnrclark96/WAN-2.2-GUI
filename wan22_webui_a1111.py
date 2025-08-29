#!/usr/bin/env python
"""Gradio-based WAN 2.2 front-end for A1111 style workflow.

This file defines a small Gradio app and a streaming runner that launches the
PowerShell shim (wan_runner.ps1). It also supports an "official" engine path
that calls the reference generate.py via the configured virtual environment
interpreter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Generator, List

from core import paths

import gradio as gr

APP_TITLE = "WAN 2.2 GUI"

SAMPLERS = [
    "unipc",
    "ddim",
    "euler",
    "euler_a",
    "heun",
    "dpmpp_2m",
    "dpmpp_2m_sde",
]

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


# PowerShell path resolver with sensible Windows fallbacks
def _powershell() -> str:
    preferred = Path(r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")
    if preferred.exists():
        return preferred.as_posix()

    import shutil

    ps = shutil.which("powershell.exe")
    if ps:
        return ps

    pwsh = shutil.which("pwsh")
    if pwsh:
        return pwsh

    return "powershell.exe"


def build_args(values: dict) -> List[str]:
    """Build the PowerShell invocation to the engine shim with CLI args."""
    args: List[str] = [_powershell(), "-NoLogo", "-File", paths.PS1_ENGINE.as_posix()]

    # Order controls how flags are emitted; keys not present are skipped.
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


def stream_run(cmd: List[str], cwd: str | None = None) -> Generator[str, None, None]:
    """
    Launch *cmd* and yield a single cumulative string so the GUI textbox
    persists the full log. We merge STDERR→STDOUT and read one stream.
    """
    import subprocess

    buf: list[str] = []
    launch = "[launch] " + " ".join(cmd)
    buf.append(launch)
    yield "\n".join(buf)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            text=True,
            bufsize=1,
        )
    except Exception as e:
        buf.append(f"[error] {e}")
        yield "\n".join(buf)
        return

    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        buf.append(line.rstrip("\r\n"))
        yield "\n".join(buf)

    code = proc.wait()
    buf.append(f"[exit] code={code}")
    yield "\n".join(buf)


def run_cmd(engine: str = "diffusers", **kw) -> Generator[str, None, None]:
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
        if not paths.PY_EXE.exists():
            raise gr.Error("Set PY_EXE path in the Paths tab.")
        if not paths.OFFICIAL_GENERATE or not paths.OFFICIAL_GENERATE.exists():
            raise gr.Error("Set OFFICIAL_GENERATE path in the Paths tab.")
        if h != 704:
            raise gr.Error("Official engine only supports height=704 (720p).")
        size = f"{w}*{h}"
        cmd: List[str] = [
            paths.PY_EXE.as_posix(),
            paths.OFFICIAL_GENERATE.as_posix(),
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
        if not paths.PS1_ENGINE.exists():
            raise gr.Error("Set PS1_ENGINE path in the Paths tab.")
        sampler = kw.get("sampler", "unipc")
        if sampler not in SAMPLERS:
            raise gr.Error(f"invalid sampler: {sampler}")
        # diffusers engine via PowerShell runner
        args = dict(kw)
        cli_mode = "t2i" if mode == "t2i" else "t2v"
        args["mode"] = cli_mode
        cmd = build_args(args)

    yield from stream_run(cmd)


def build_ui():
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"## {APP_TITLE}", elem_id="app-title")

        with gr.Tabs():
            with gr.Tab("Generate"):
                engine = gr.Radio(["diffusers", "official"], value="diffusers", label="Engine")
                mode = gr.Radio(["T2V", "T2I"], value="T2V", label="Generation Mode")

                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=3)
                    neg_prompt = gr.Textbox(label="Negative Prompt", lines=2)

                with gr.Row():
                    sampler = gr.Dropdown(
                        SAMPLERS,
                        value="unipc",
                        label="Sampler",
                        info="Only Diffusers schedulers validated with WAN.",
                    )
                    steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    cfg = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG")
                    seed = gr.Number(value=-1, label="Seed")

                sampler_note = gr.Markdown("Sampler is a Diffusers-only setting.", visible=False)

                with gr.Row():
                    fps = gr.Slider(1, 30, value=24, step=1, label="FPS")
                    frames = gr.Slider(1, 97, value=16, step=1, label="Frames")
                    width = gr.Number(value=1280, label="Width")
                    height = gr.Number(value=704, label="Height")

                with gr.Row():
                    batch_count = gr.Number(value=1, label="Batch Count")
                    batch_size = gr.Number(value=1, label="Batch Size")
                    dtype = gr.Dropdown(["fp16", "bf16", "fp32"], value="bf16", label="DType")
                    attn = gr.Dropdown(["sdpa", "flash-attn"], value="flash-attn", label="Attention")

                with gr.Row():
                    model_dir = gr.Textbox(value=DEFAULT_MODEL_DIR, label="Model Dir")
                    outdir = gr.Textbox(value=DEFAULT_OUTDIR, label="Output Dir")

                image = gr.Image(label="Init Image", type="filepath")

                run = gr.Button("Generate")
                log = gr.Textbox(label="Log", lines=18, show_copy_button=True)

            with gr.Tab("Paths"):
                py_exe = gr.Textbox(paths.PY_EXE.as_posix(), label="PY_EXE")
                ps1_engine = gr.Textbox(paths.PS1_ENGINE.as_posix(), label="PS1_ENGINE")
                official = gr.Textbox(paths.OFFICIAL_GENERATE.as_posix(), label="OFFICIAL_GENERATE")
                save_paths = gr.Button("Save Paths")

                def on_save(py: str, ps1: str, off: str) -> None:
                    paths.save_config({
                        "PY_EXE": py,
                        "PS1_ENGINE": ps1,
                        "OFFICIAL_GENERATE": off,
                    })
                    paths.PY_EXE = Path(py)
                    paths.PS1_ENGINE = Path(ps1)
                    paths.OFFICIAL_GENERATE = Path(off)
                    gr.Info("paths saved")

                save_paths.click(on_save, inputs=[py_exe, ps1_engine, official], outputs=[])

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
            mode_sel,
        ):
            sel = str(mode_sel or "T2V").upper()
            if sel == "T2I":
                mode_v = "t2i"
            else:
                mode_v = "ti2v" if (prompt_v and image_v) else ("i2v" if image_v else "t2v")

            # Official engine does not support pure image-to-video in this GUI
            if eng == "official" and mode_v == "i2v":
                raise gr.Error("Use the diffusers engine for image-to-video.")

            gr.Info(f"mode={mode_v}")

            # Sanitize numeric inputs
            steps_v = safe_int(steps_v, 20, 1)
            cfg_v = safe_float(cfg_v, 7.0, 0.0)
            fps_v = safe_int(fps_v, 24, 1)
            if mode_v == "t2i":
                frames_v = 1
            else:
                frames_v = safe_int(frames_v, 16, 1)
                if (frames_v - 1) % 4 != 0:
                    frames_v = (frames_v - 1) // 4 * 4 + 1
            orig_w = safe_int(width_v, 1280, 32)
            orig_h = safe_int(height_v, 704, 32)
            width_v = snap32(orig_w)
            height_v = snap32(orig_h)
            if width_v != orig_w:
                gr.Warning(f"Snapped {orig_w} → {width_v}")
            if height_v != orig_h:
                gr.Warning(f"Snapped {orig_h} → {height_v}")
            batch_count_v = safe_int(batch_count_v, 1, 1)
            batch_size_v = safe_int(batch_size_v, 1, 1)
            try:
                seed_v = int(seed_v)
            except Exception:
                seed_v = -1

            run_kw = {
                "mode": mode_v,
                "prompt": str(prompt_v or ""),
                "neg_prompt": str(neg_v or ""),
                "steps": steps_v,
                "cfg": cfg_v,
                "seed": seed_v,
                "fps": fps_v,
                "frames": frames_v,
                "width": width_v,
                "height": height_v,
                "batch_count": batch_count_v,
                "batch_size": batch_size_v,
                "outdir": outdir_v,
                "model_dir": model_dir_v,
                "dtype": dtype_v,
                "attn": attn_v,
                "image": image_v,
            }

            if eng == "diffusers":
                run_kw["sampler"] = sampler_v

            for line in run_cmd(eng, **run_kw):
                yield line

        # Bind UI actions
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
                mode,
            ],
            outputs=log,
        )

        engine.change(
            lambda e: (
                gr.Dropdown.update(interactive=(e == "diffusers")),
                gr.Markdown.update(visible=(e == "official")),
            ),
            inputs=[engine],
            outputs=[sampler, sampler_note],
        )

        def _on_mode_change(m: str, cur_frames: int):
            is_t2i = (m or "T2V").upper() == "T2I"
            frames_val = 1 if is_t2i else max(1, cur_frames)
            return (
                gr.update(value=frames_val, interactive=not is_t2i),
                gr.update(interactive=not is_t2i),
            )

        mode.change(
            _on_mode_change,
            inputs=[mode, frames],
            outputs=[frames, fps],
            queue=False,
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
