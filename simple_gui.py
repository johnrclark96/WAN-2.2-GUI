import subprocess
import time
import re
from pathlib import Path
import gradio as gr

# Paths
HERE = Path(__file__).parent.resolve()
RUNNER = HERE / "run_wan22.py"
MODEL = HERE / "models" / "TI2V_5B"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)
TIMEOUT = 600  # seconds

# Sanity checks for essential files
if not RUNNER.exists():
    raise FileNotFoundError(f"Missing runner script: {RUNNER}")
if not MODEL.exists():
    raise FileNotFoundError(f"Missing model directory: {MODEL}")
if not (MODEL / "model_index.json").exists():
    raise FileNotFoundError(
        f"Model path {MODEL} does not look like a Diffusers checkpoint"
    )


def generate(
    prompt,
    neg="",
    steps=20,
    width=576,
    height=320,
    fps=24,
    frames=48,
    seed=-1,
    progress=gr.Progress(track_tqdm=True),
):
    """Run the WAN runner and stream its logs and latest video with progress."""
    cmd = [
        "python",
        str(RUNNER),
        "--mode",
        "ti2v",
        "--prompt",
        prompt,
        "--steps",
        str(int(steps)),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--fps",
        str(int(fps)),
        "--frames",
        str(int(frames)),
        "--seed",
        str(int(seed)),
        "--outdir",
        str(OUTDIR),
        "--base",
        str(MODEL),
    ]
    if neg and neg.strip():
        cmd += ["--neg_prompt", neg]

    logs = ""
    start_time = time.time()
    progress(0.0, desc="Starting")
    yield logs, None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logs += line
            # Detect progress lines like: [PROGRESS] ... percent=42
            m = re.search(r"percent=(\d+)", line)
            if m:
                pct = int(m.group(1))
                progress(pct / 100.0, desc=f"Generating {pct}%")
            if "Loading model" in line:
                progress(0.0, desc="Importing model")
            yield logs, None
        rc = proc.wait()
        videos = sorted(
            (p for p in OUTDIR.glob("*.mp4") if p.stat().st_mtime > start_time),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        video = str(videos[0]) if videos else None
        if rc != 0 or not video:
            if rc != 0:
                logs += f"\n[ERROR] WAN runner exited with code {rc}"
            if not video:
                logs += "\n[ERROR] No fresh video produced."
            progress(0, desc="Error")
            yield logs, None
            return
        progress(1.0, desc="Done")
        yield logs, video
    except Exception as e:
        logs += f"\n[EXCEPTION] {e}"
        progress(0, desc="Error")
        yield logs, None


with gr.Blocks(title="WAN 2.2 Lite") as demo:
    gr.Markdown("### WAN 2.2 – lightweight GUI")
    prompt = gr.Textbox(label="Prompt")
    neg = gr.Textbox(label="Negative prompt", value="")
    steps = gr.Slider(1, 80, 20, step=1, label="Steps")
    width = gr.Number(value=576, label="Width")
    height = gr.Number(value=320, label="Height")
    fps = gr.Number(value=24, label="FPS")
    frames = gr.Number(value=48, label="Frames")
    seed = gr.Number(value=-1, label="Seed (-1=random)")
    gen = gr.Button("Generate", variant="primary")
    logs = gr.Textbox(label="Logs")
    video = gr.Video(label="Result")
    gen.click(generate, [prompt, neg, steps, width, height, fps, frames, seed], [logs, video])

if __name__ == "__main__":
    demo.launch()
