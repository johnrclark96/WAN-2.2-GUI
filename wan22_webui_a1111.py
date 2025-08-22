#!/usr/bin/env python
# wan22_webui_a1111.py – A1111-style Gradio UI for WAN 2.2

import os, re, sys, json, time, shutil, random, argparse, subprocess, shlex, socket
from pathlib import Path
from typing import List

try:
    import gradio as gr
except Exception:
    # If Gradio isn't installed in this venv, exit with message
    raise SystemExit("Gradio is required. Install with: pip install gradio")

APP_TITLE = "WAN 2.2 – A1111-style UI"
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_RUNNER = (THIS_DIR / "run_wan22.py").as_posix()
DEFAULT_OUT    = (THIS_DIR / "outputs").as_posix()
DEFAULT_LORA_DIR = (THIS_DIR / "loras")
DEFAULT_LORA_DIR.mkdir(parents=True, exist_ok=True)
Path(DEFAULT_OUT).mkdir(parents=True, exist_ok=True)

# Detect available WAN model(s) in the models folder for selection
MODEL_DIR = THIS_DIR / "models"
model_paths = []
if MODEL_DIR.exists():
    for p in MODEL_DIR.iterdir():
        if p.is_dir() and (p/"model_index.json").exists():
            model_paths.append(str(p))
model_paths.sort()
model_choices = []
if len(model_paths) <= 1:
    model_choices = ["Auto"] + model_paths
    default_model = "Auto"
else:
    model_choices = model_paths
    default_model = model_paths[0]

# ---------- LoRA weight parsing ----------
WEIGHT_PATTERNS = [
    re.compile(r'@(?P<w>\d(?:\.\d+)?)\.(?:safetensors|pt)$', re.I),
    re.compile(r'\((?P<w>\d(?:\.\d+)?)\)\.(?:safetensors|pt)$', re.I),
    re.compile(r'[-_~]w(?P<w>\d(?:\.\d+)?)\.(?:safetensors|pt)$', re.I),
]
def parse_weight_from_name(name: str, default=0.8) -> float:
    """Extract weight from LoRA filename by common patterns."""
    for rgx in WEIGHT_PATTERNS:
        m = rgx.search(name)
        if m:
            try:
                w = float(m.group("w"))
                return max(0.0, min(2.0, w))
            except:
                pass
    return default

def normalize_lora_table(lora_rows):
    """
    Accepts various formats (list-of-lists, list-of-dicts, DataFrame).
    Returns list of [path, weight] pairs.
    """
    if lora_rows is None:
        return []
    try:
        import pandas as pd
        if isinstance(lora_rows, pd.DataFrame):
            return lora_rows.to_numpy().tolist()
    except Exception:
        pass
    # Gradio sometimes gives dict {"data": [...]}
    if isinstance(lora_rows, dict) and "data" in lora_rows:
        return lora_rows["data"]
    # Normalize each row to [path, weight]
    rows = []
    for r in lora_rows:
        if isinstance(r, (list, tuple)):
            rows.append(list(r))
        elif isinstance(r, dict):
            rows.append([
                r.get("Path") or r.get("path") or r.get(0) or "",
                r.get("Weight") or r.get("weight") or r.get(1) or 0.8,
            ])
        else:
            rows.append([str(r), 0.8])
    return rows

def ingest_loras(files, lora_dir: Path) -> List[List]:
    """
    Copy uploaded LoRA files into lora_dir. 
    Return table rows [[path, weight], ...] for each valid LoRA.
    """
    rows = []
    lora_dir.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        src = Path(f if isinstance(f, (str, Path)) else getattr(f, "name", ""))
        if not src.name:
            continue
        if src.suffix.lower() not in (".safetensors", ".pt"):
            continue  # skip non-LoRA files
        dst = lora_dir / src.name
        try:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            rows.append([dst.as_posix(), parse_weight_from_name(dst.name)])
        except OSError as e:
            gr.Warning(f"Failed to ingest {src.name}: {e}")
            continue
    return rows

# ---------- Command builder ----------
def build_cmd(
    runner, mode, prompt, neg, init_img, sampler, steps, cfg, seed,
    fps, frames, width, height, batch_count, batch_size, lora_rows, outdir, extra
) -> List[str]:
    """Build the command list to run the WAN runner script with given parameters."""
    if not runner or not Path(runner).exists():
        raise gr.Error(f"Runner not found: {runner}. Set a valid path in 'Runner / Output'.")
    if mode == "t2v" and not (prompt or "").strip():
        raise gr.Error("Prompt is required for txt2vid.")
    if mode in ("i2v", "ti2v") and init_img:
        if not Path(init_img).exists():
            raise gr.Error(f"Init image not found: {init_img}")

    cmd = [sys.executable, str(Path(runner).resolve()), "--mode", mode, "--prompt", prompt or ""]
    if neg:
        cmd += ["--neg_prompt", neg]
    if sampler:
        cmd += ["--sampler", sampler]
    if steps:
        cmd += ["--steps", str(int(steps))]
    if cfg is not None:
        cmd += ["--cfg", str(float(cfg))]
    if seed is not None and str(seed) != "":
        cmd += ["--seed", str(int(seed))]
    if fps:
        cmd += ["--fps", str(int(fps))]
    if frames:
        cmd += ["--frames", str(int(frames))]
    if width:
        cmd += ["--width", str(int(width))]
    if height:
        cmd += ["--height", str(int(height))]
    if batch_count:
        cmd += ["--batch_count", str(int(batch_count))]
    if batch_size:
        cmd += ["--batch_size", str(int(batch_size))]
    if outdir:
        cmd += ["--outdir", outdir]
    if mode in ("i2v", "ti2v") and init_img:
        cmd += ["--init_image", init_img]

    # Attach LoRA arguments
    rows = normalize_lora_table(lora_rows)
    for row in rows:
        path = str(row[0]).strip() if row and row[0] else ""
        if not path:
            continue
        try:
            w = float(row[1]) if len(row) > 1 else 0.8
        except:
            w = 0.8
        cmd += ["--lora", f"{path}:{w}"]

    # Extra arguments (including --base if added)
    if extra:
        try:
            cmd += shlex.split(extra)
        except Exception:
            cmd += [extra]  # fallback: treat as single token

    return cmd

# ---------- Subprocess runner ----------
PROC = None
def stream_run(cmd: List[str], outdir: Path, progress=gr.Progress(track_tqdm=True)):
    """
    Run the generation subprocess and yield (logs, video_path, info_dict) for streaming.
    """
    global PROC
    logs = ""
    video = None
    info = {"command": " ".join(cmd)}
    start_time = time.time()
    had_error = False

    progress(0.0, desc="Starting")
    # Initial yield (empty) to refresh UI immediately
    yield logs, None, info

    try:
        PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=str(THIS_DIR),  # run in D:\wan22 directory
        )
        assert PROC.stdout is not None
        for line in PROC.stdout:
            logs += line
            # Update progress based on runner messages
            if "Loading model" in line:
                progress(0.0, desc="Importing model")
            m_pct = re.search(r"percent=(\d+)", line)
            if m_pct:
                pct = int(m_pct.group(1))
                progress(pct / 100.0, desc=f"Generating {pct}%")
            # Detect a saved output path in the engine logs:
            m = re.search(r"(saved|wrote|output)[:\s]+(.+\.(?:mp4|gif|webm|mov))", line, re.I)
            if m and not video:
                p = Path(m.group(2).strip().strip('"'))
                if not p.is_absolute():
                    p = (outdir / p).resolve()
                if p.exists():
                    video = p.as_posix()
            yield logs, video, info
        rc = PROC.wait()
        info["return_code"] = rc
        logs += f"\n[Exit code] {rc}\n"
        if rc != 0:
            had_error = True
    except FileNotFoundError as e:
        logs += f"\n[ERROR] {e}\nCheck the 'Runner path' and ensure the Python script exists.\n"
        had_error = True
        progress(0, desc="Error")
    except Exception as e:
        logs += f"\n[ERROR] {e}\n"
        had_error = True
        progress(0, desc="Error")
    finally:
        # If no video caught in logs, try finding the newest file in outdir as fallback
        if not video and outdir.exists():
            newest, latest_time = None, 0
            for ext in (".mp4", ".gif", ".webm", ".mov"):
                for f in outdir.rglob(f"*{ext}"):
                    try:
                        t = f.stat().st_mtime
                    except:
                        continue
                    if t >= start_time and t > latest_time:
                        newest, latest_time = f, t
            if newest:
                video = newest.as_posix()
        PROC = None
        if had_error or not video:
            progress(0, desc="Error")
        else:
            progress(1.0, desc="Done")
        yield logs, video, info

def interrupt_proc():
    """Terminate the running generation, if any."""
    global PROC
    if PROC:
        try:
            if PROC.poll() is None:
                PROC.terminate()
                try:
                    PROC.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    PROC.kill()
                    PROC.wait()
                return "Interrupted."
            else:
                return "No active process."
        except Exception as e:
            return f"Failed to interrupt: {e}"
        finally:
            PROC = None
    return "No active process."

# ---------- Build UI ----------
CSS = """
.gradio-container {max-width: 1500px !important;}
#app-title h1 { font-weight: 700; }
:root { --button-primary-background-fill: #f39c12; --button-primary-text-color: #111; }
#left-col { background: #1f2125; border-radius: 10px; padding: 12px; }
#right-col{ background: #1b1d21; border-radius: 10px; padding: 12px; }
#generate-btn { height: 64px; font-size: 18px; font-weight: 700; }
#interrupt-btn { height: 40px; }
.smallbtn { height: 40px; }
"""

# Preset negative prompts for styles (optional usage)
STYLES = {
    "None": {"neg": ""},
    "Cinematic Night": {"neg": "jitter, flicker, washed colors, dull contrast, artifacts"},
    "Clean Daylight": {"neg": "harsh shadows, overexposed, banding, compression artifacts"},
    "Smooth Portrait": {"neg": "blur, off-face distortions, frame warping"},
}

SAMPLERS = ["ddim", "dpmpp_2m_sde", "dpmpp_3m_sde", "unipc", "euler_a"]

def build_ui():
    with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"## {APP_TITLE}", elem_id="app-title")

        with gr.Tabs():
            # ===================== TXT2VID TAB =====================
            with gr.Tab("txt2vid"):
                with gr.Row():
                    with gr.Column(scale=6, elem_id="left-col"):
                        with gr.Row():
                            model_sel = gr.Dropdown(label="WAN Checkpoint", choices=model_choices, value=default_model)
                            style_dd = gr.Dropdown(choices=list(STYLES.keys()), value="None", label="Styles")

                        with gr.Row():
                            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your scene...")
                            with gr.Column(scale=1):
                                generate = gr.Button("Generate", variant="primary", elem_id="generate-btn")
                                interrupt = gr.Button("⏹️ Stop", elem_id="interrupt-btn")
                                send_to_img = gr.Button("Send to img2vid", elem_classes=["smallbtn"])

                        neg = gr.Textbox(label="Negative prompt (Alt+Enter to generate)", lines=2)

                        with gr.Row():
                            sampler = gr.Dropdown(choices=SAMPLERS, value="dpmpp_3m_sde", label="Sampling method")
                            steps = gr.Slider(1, 80, value=20, step=1, label="Sampling steps")

                        with gr.Row():
                            width = gr.Slider(320, 1920, value=1024, step=64, label="Width")
                            height = gr.Slider(240, 1080, value=576, step=16, label="Height")

                        with gr.Row():
                            fps = gr.Slider(1, 60, value=24, step=1, label="FPS")
                            frames = gr.Slider(8, 240, value=48, step=1, label="Frames")

                        with gr.Row():
                            batch_count = gr.Slider(1, 8, value=1, step=1, label="Batch count")
                            batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch size")
                            cfg = gr.Slider(0.0, 15.0, value=7.0, step=0.5, label="CFG Scale")

                        with gr.Row():
                            seed = gr.Textbox(label="Seed (-1 = random)", value="-1")
                            dice = gr.Button("🎲", elem_classes=["smallbtn"])
                            recycle = gr.Button("♻️", elem_classes=["smallbtn"])  # (Recycle button can reuse last seed if desired)

                        gr.Markdown("**LoRAs** (drop files or add paths below):")
                        t_files = gr.Files(label="LoRA files (.safetensors/.pt)", file_count="multiple", type="filepath")
                        t_loras = gr.Dataframe(headers=["Path", "Weight"], datatype=["str", "number"], row_count=(0, "dynamic"))
                        add_lora_btn = gr.Button("Add uploaded LoRAs to table")

                        with gr.Accordion("Runner / Output", open=False):
                            runner = gr.Textbox(label="Runner path", value=DEFAULT_RUNNER)
                            outdir = gr.Textbox(label="Output directory", value=DEFAULT_OUT)
                            extra = gr.Textbox(label="Extra args", placeholder="--vae path\\to\\vae.safetensors (etc.)")

                    with gr.Column(scale=5, elem_id="right-col"):
                        result_video = gr.Video(label="Result")
                        gen_info = gr.JSON(label="Generation Info")
                        console = gr.Textbox(label="Console Output", lines=20)

                # Event handlers for txt2vid tab
                def on_style(style, current_neg):
                    # When a style preset is selected, update the negative prompt
                    return STYLES.get(style, {}).get("neg", current_neg)
                style_dd.change(on_style, inputs=[style_dd, neg], outputs=[neg])

                def on_add_loras(files, current_table):
                    existing = normalize_lora_table(current_table)
                    new_rows = ingest_loras(files, DEFAULT_LORA_DIR)
                    return existing + new_rows
                add_lora_btn.click(on_add_loras, inputs=[t_files, t_loras], outputs=[t_loras])

                def make_seed():
                    return str(random.randint(1, 2**31 - 1))
                dice.click(make_seed, inputs=[], outputs=[seed])

                def send_txt_to_img(p, n):
                    # Send prompt/negatives from txt2vid to img2vid tab
                    return p, n
                send_to_img.click(send_txt_to_img, inputs=[prompt, neg], outputs=[])

                def do_generate_txt(p, n, samp, st, w, h, f_fps, f_frames, bcnt, bsz, cfg_s, seed_s, loras_tbl, model_choice, runner_p, out_dir, extra_flags, progress=gr.Progress(track_tqdm=True)):
                    # Validate early
                    if not (p or "").strip():
                        raise gr.Error("Prompt is required for txt2vid.")
                    if not Path(runner_p).exists():
                        raise gr.Error(f"Runner not found: {runner_p}")

                    # Normalize seed to int (or random if -1/invalid)
                    try:
                        s = int(seed_s)
                        if s < 0:
                            s = random.randint(1, 2**31 - 1)
                    except Exception:
                        s = random.randint(1, 2**31 - 1)

                    # Insert model selection into extra args if needed
                    base_path = model_choice
                    if base_path and str(base_path).strip().lower() != "auto":
                        extra_flags = (extra_flags or "").strip()
                        extra_flags = f"--base \"{str(base_path).strip()}\" " + extra_flags

                    # Build command list and run
                    cmd = build_cmd(
                        runner=runner_p, mode="t2v", prompt=p, neg=n, init_img=None,
                        sampler=samp, steps=st, cfg=cfg_s, seed=s,
                        fps=f_fps, frames=f_frames, width=w, height=h,
                        batch_count=bcnt, batch_size=bsz,
                        lora_rows=loras_tbl, outdir=out_dir, extra=extra_flags
                    )
                    # Stream output (this function yields incremental results for console/video)
                    yield from stream_run(cmd, Path(out_dir), progress)

                # Connect generate button (txt2vid)
                generate.click(
                    do_generate_txt,
                    inputs=[prompt, neg, sampler, steps, width, height, fps, frames, batch_count, batch_size, cfg, seed, t_loras, model_sel, runner, outdir, extra],
                    outputs=[console, result_video, gen_info]
                )
                interrupt.click(lambda: interrupt_proc(), inputs=[], outputs=[console])

            # ===================== IMG2VID TAB =====================
            with gr.Tab("img2vid"):
                with gr.Row():
                    with gr.Column(scale=6, elem_id="left-col"):
                        with gr.Row():
                            model_sel2 = gr.Dropdown(label="WAN Checkpoint", choices=model_choices, value=default_model)
                            style_dd2 = gr.Dropdown(choices=list(STYLES.keys()), value="None", label="Styles")

                        with gr.Row():
                            prompt2 = gr.Textbox(label="Prompt", lines=4, placeholder="Prompt (optional if init image provided)")
                            with gr.Column(scale=1):
                                generate2 = gr.Button("Generate", variant="primary", elem_id="generate-btn")
                                interrupt2 = gr.Button("⏹️ Stop", elem_id="interrupt-btn")
                                send_to_txt = gr.Button("Send to txt2vid", elem_classes=["smallbtn"])

                        neg2 = gr.Textbox(label="Negative prompt", lines=2)
                        init_img = gr.Image(label="Init Image", type="filepath")

                        with gr.Row():
                            sampler2 = gr.Dropdown(choices=SAMPLERS, value="dpmpp_3m_sde", label="Sampling method")
                            steps2 = gr.Slider(1, 80, value=20, step=1, label="Sampling steps")

                        with gr.Row():
                            width2 = gr.Slider(320, 1920, value=1024, step=64, label="Width")
                            height2 = gr.Slider(240, 1080, value=576, step=16, label="Height")

                        with gr.Row():
                            fps2 = gr.Slider(1, 60, value=24, step=1, label="FPS")
                            frames2 = gr.Slider(8, 240, value=48, step=1, label="Frames")

                        with gr.Row():
                            batch_count2 = gr.Slider(1, 8, value=1, step=1, label="Batch count")
                            batch_size2 = gr.Slider(1, 8, value=1, step=1, label="Batch size")
                            cfg2 = gr.Slider(0.0, 15.0, value=7.0, step=0.5, label="CFG Scale")

                        with gr.Row():
                            seed2 = gr.Textbox(label="Seed (-1 = random)", value="-1")
                            dice2 = gr.Button("🎲", elem_classes=["smallbtn"])

                        gr.Markdown("**LoRAs** (drop files or add paths below):")
                        i_files = gr.Files(label="LoRA files (.safetensors/.pt)", file_count="multiple", type="filepath")
                        i_loras = gr.Dataframe(headers=["Path", "Weight"], datatype=["str", "number"], row_count=(0, "dynamic"))
                        add_lora_btn2 = gr.Button("Add uploaded LoRAs to table")

                        with gr.Accordion("Runner / Output", open=False):
                            runner2 = gr.Textbox(label="Runner path", value=DEFAULT_RUNNER)
                            outdir2 = gr.Textbox(label="Output directory", value=DEFAULT_OUT)
                            extra2 = gr.Textbox(label="Extra args")

                    with gr.Column(scale=5, elem_id="right-col"):
                        result_video2 = gr.Video(label="Result")
                        gen_info2 = gr.JSON(label="Generation Info")
                        console2 = gr.Textbox(label="Console Output", lines=20)

                # Event handlers for img2vid tab
                style_dd2.change(on_style2 := (lambda style, current_neg: STYLES.get(style, {}).get("neg", current_neg)), 
                                 inputs=[style_dd2, neg2], outputs=[neg2])
                add_lora_btn2.click(lambda files, table: normalize_lora_table(table) + ingest_loras(files, DEFAULT_LORA_DIR),
                                    inputs=[i_files, i_loras], outputs=[i_loras])
                dice2.click(make_seed, inputs=[], outputs=[seed2])

                def send_img_to_txt(p, n):
                    return p, n
                send_to_txt.click(send_img_to_txt, inputs=[prompt2, neg2], outputs=[])

                def do_generate_img(p, n, init, samp, st, w, h, f_fps, f_frames, bcnt, bsz, cfg_s, seed_s, loras_tbl, model_choice2, runner_p, out_dir, extra_flags, progress=gr.Progress(track_tqdm=True)):
                    # Validate inputs
                    if not init and not (p or "").strip():
                        raise gr.Error("Provide an init image and/or a prompt.")
                    if not Path(runner_p).exists():
                        raise gr.Error(f"Runner not found: {runner_p}")
                    if init and not Path(init).exists():
                        raise gr.Error(f"Init image not found: {init}")

                    try:
                        s = int(seed_s)
                        if s < 0:
                            s = random.randint(1, 2**31 - 1)
                    except Exception:
                        s = random.randint(1, 2**31 - 1)

                    # Determine mode: if prompt given, use text+image (ti2v), otherwise image-to-video (i2v)
                    mode = "ti2v" if (p or "").strip() else "i2v"

                    # Insert model selection into extra args if needed
                    base_path = model_choice2
                    if base_path and str(base_path).strip().lower() != "auto":
                        extra_flags = (extra_flags or "").strip()
                        extra_flags = f"--base \"{str(base_path).strip()}\" " + extra_flags

                    cmd = build_cmd(
                        runner=runner_p, mode=mode, prompt=p or "", neg=n, init_img=init,
                        sampler=samp, steps=st, cfg=cfg_s, seed=s,
                        fps=f_fps, frames=f_frames, width=w, height=h,
                        batch_count=bcnt, batch_size=bsz,
                        lora_rows=loras_tbl, outdir=out_dir, extra=extra_flags
                    )
                    yield from stream_run(cmd, Path(out_dir), progress)

                generate2.click(
                    do_generate_img,
                    inputs=[prompt2, neg2, init_img, sampler2, steps2, width2, height2, fps2, frames2, batch_count2, batch_size2, cfg2, seed2, i_loras, model_sel2, runner2, outdir2, extra2],
                    outputs=[console2, result_video2, gen_info2]
                )
                interrupt2.click(lambda: interrupt_proc(), inputs=[], outputs=[console2])

            # ===================== CONSOLE TAB =====================
            with gr.Tab("Console"):
                gr.Markdown("Use the console outputs in each tab to monitor generation. This tab is a placeholder for future global logs.")

            # ===================== SETTINGS TAB =====================
            with gr.Tab("Settings"):
                gr.Markdown("Preset styles JSON (editable):")
                style_json = gr.JSON(value=STYLES, label="Styles JSON")

                def update_styles(js):
                    # Overwrite STYLES dict (in-memory) with user edits
                    try:
                        return js
                    except Exception:
                        return STYLES

                style_json.change(update_styles, inputs=[style_json], outputs=[style_json])

    return demo

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--listen", action="store_true", help="Allow other PCs to access, bind to 0.0.0.0")
    ap.add_argument("--auth", type=str, default=None, help="Optional basic auth (user:pass)")
    args = ap.parse_args()

    ui = build_ui()
    auth = tuple(args.auth.split(":", 1)) if args.auth and ":" in args.auth else None

    # Auto-switch port if 7860 is busy (e.g., if Stable Diffusion WebUI is running)
    chosen_port = args.port
    if args.port == 7860 and is_port_in_use(7860):
        if not is_port_in_use(7861):
            print("[ui] Port 7860 busy; switching to 7861.", flush=True)
            chosen_port = 7861
        else:
            print("[ui] Ports 7860 and 7861 are busy. Using 7860 and letting Gradio handle the conflict.", flush=True)

    ui.launch(
        server_name=("0.0.0.0" if args.listen else "127.0.0.1"),
        server_port=chosen_port,
        auth=auth,
        inbrowser=False,
        show_api=False,
    )

if __name__ == "__main__":
    main()
