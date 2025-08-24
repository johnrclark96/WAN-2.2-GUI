Purpose
This file tells the Codex agent exactly how to work with this repository from a Linux container, while the project itself targets Windows 11 with WAN 2.2 (5B only) at D:\wan22.

Project truths (do not change)
• WAN-only, 5B model only, no 14B paths anywhere.
• All runtime paths are under D:\wan22 on Windows (venv, models, outputs).
• A1111 naming is allowed as UI style only; there is no integration with other tools.
• Windows CI validates the PowerShell runner and any Windows-specific behavior.
• Codex must never attempt WAN inference or model downloads.

What the agent may assume about its container
• It runs on Linux in the codex-universal image.
• pip/ruff/mypy/pytest are already installed via environment Setup/Maintenance (human-configured).
• Internet during tasks may be limited; do not rely on live installs.
• Do not modify the container or ask to change Setup/Maintenance scripts.

Command palette for Codex (run these in the container)
• ruff check .
• mypy --ignore-missing-imports wan_ps1_engine.py core
• python -m compileall -q .
• python wan_ps1_engine.py --help
• python wan_ps1_engine.py --dry-run --mode t2v --prompt ok --frames 4 --fps 24 --width 1280 --height 704 --outdir out
• pytest -q -k "not test_runner_path" (skip the Windows PowerShell runner test in this container)

Never do in the container
• Do not run WAN generation.
• Do not download models or install heavy libraries (torch, diffusers, flash-attn, transformers, accelerate, imageio-ffmpeg).
• Do not attempt to run PowerShell.

File map (focus when editing/auditing)
• wan_ps1_engine.py — main CLI, input validation, attention backend, result protocol, dry-run.
• core/wan_video.py — T2V/I2V/TI2V flow via 5B pipeline.
• core/wan_image.py — T2I single-frame (forces frames=1, saves PNG).
• wan_runner.ps1 — Windows launcher; resolves D:\wan22\venv\Scripts\python.exe; prints “[WAN shim] Launch: …”.
• tests/ — keep all tests runnable in Linux except test_runner_path (Windows-only).
• .github/workflows/wan-ci.yml — Windows-only CI.

Editing rules for the agent

Keep dry-run model-free and early-exit. In wan_ps1_engine.py, --dry-run must return before any heavy imports or model_dir checks, print exactly one “[RESULT] OK …” line, and exit 0.

Attention backend must use the modern PyTorch API. Replace any use of torch.backends.cuda.sdp_kernel with torch.nn.attention.sdpa_kernel (updated signature). Do not print the deprecated symbol.

Enforce 5B defaults. If --model_dir is omitted, use D:\wan22\models\Wan2.2-TI2V-5B-Diffusers. Do not include any 14B logic.

Routing: one model, two behaviors. ti2v-5B is the task; presence of an image means I2V, otherwise T2V. Mode t2i is a thin convenience that forces frames=1 and writes a PNG.

Resolution policy. For 720p presets, snap to 1280×704 or 704×1280 (not 1280×720). For custom sizes, snap width and height to multiples of 32 and log the snapped size.

Validation before heavy imports. t2v/t2i require nonempty prompt; i2v/ti2v require existing image path; steps>=1; frames>=1; warn (non-fatal) when res > 1280×704 or frames > 64 on 16 GB.

Result protocol (CI/GUI rely on this). On success: print one “[OUTPUT] <absolute path>” and one “[RESULT] OK …” line; write a JSON sidecar next to the media. On failure: print one “[RESULT] FAIL GENERATION …” line with error type/message/trace, and write the same JSON sidecar.

VRAM hygiene. Prefer bf16/fp16 for pipeline; VAE decode in fp32 if needed; enable model CPU offload hooks; in finally, delete large objects and call torch.cuda.empty_cache().

Windows CI (what the workflow must do; agent edits YAML, the runner executes)
• Single job on windows-latest using pwsh.
• Steps order: checkout → setup Python 3.10 → pip install ruff mypy pytest → ruff → mypy (engine/core) → compileall → engine --help → engine --dry-run (no --model_dir) → pytest -q.
• test_runner_path runs here and must invoke wan_runner.ps1 via pwsh -NoLogo -ExecutionPolicy Bypass -File … and assert the “[WAN shim] Launch:” marker.
• No Ubuntu jobs; no denylist/grep gates.

Local Windows usage (for humans, not for Codex)
• WAN venv: D:\wan22\venv
• Models: D:\wan22\models\Wan2.2-TI2V-5B-Diffusers
• Outputs: D:\wan22\outputs
• Example T2V (real run): choose safe defaults (e.g., 1280×704, 24 fps, 24–48 frames) and offload flags as documented; T2I sets frames=1 automatically.

PR checklist for the agent (include in each PR description)
• Dry-run returns early and prints a single “[RESULT] OK …” line.
• No references to torch.backends.cuda.sdp_kernel remain; attention uses sdpa_kernel.
• Defaults to 5B path when --model_dir omitted.
• T2I writes PNG (+ sidecar); T2V writes video; both print [OUTPUT] and [RESULT].
• Runner prints “[WAN shim] Launch: …” and resolves D:\wan22\venv\Scripts\python.exe.
• README/docs reflect D:\wan22 layout, 5B-only, and model-free dry-run.
• CI: windows-latest only; steps pass; no Ubuntu jobs or grep-based gates.

Troubleshooting notes for the agent
• If a command requires PowerShell or Windows paths, skip it here; ensure it’s covered in Windows CI.
• If ruff/mypy/pytest are missing, do not install them yourself; alert in the PR comment that the environment Setup/Maintenance needs to be rebuilt by the human (Reset cache).
• If dry-run attempts to import torch/diffusers or read models, move the dry-run short-circuit earlier in wan_ps1_engine.py.
