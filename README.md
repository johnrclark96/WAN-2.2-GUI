# WAN 2.2 GUI

Minimal runner and GUI for the WAN 2.2 video generation pipelines.

## Quick start (Windows)

The engine expects a virtual environment at `D:\wan22\venv`, models in
`D:\wan22\models\Wan2.2-TI2V-5B-Diffusers`, and outputs written to
`D:\wan22\outputs`.

Run the lightweight help and dry-run commands to verify the setup:

```powershell
python wan_ps1_engine.py --help
python wan_ps1_engine.py --dry-run --mode t2v --prompt ok --frames 8 --fps 24 --width 1280 --height 704
```

With models installed the same CLI can generate media:

```powershell
# text to video
python wan_ps1_engine.py --mode t2v --prompt "sunrise" --frames 8 --fps 24 --width 1280 --height 704

# text to image (frames=1 yields a PNG)
python wan_ps1_engine.py --mode t2i --prompt "ok one frame" --frames 1 --width 1280 --height 704
```

See [docs/env.md](docs/env.md) for pinned dependency versions and installation notes.
The file `wan22_frozen_requirements.txt` lists the expected package versions.
Run `python check_env.py` to verify that the current environment matches these pins.

See [CHANGELOG.md](CHANGELOG.md) for recent updates.

## Sampler (Diffusers only)

The GUI exposes a validated subset of Diffusers schedulers:

- `unipc` – UniPC multistep scheduler.
- `ddim` – DDIM scheduler.
- `euler` – Euler scheduler.
- `euler_a` – Euler ancestral scheduler.
- `heun` – Heun scheduler.
- `dpmpp_2m` – DPM++ 2M scheduler.
- `dpmpp_2m_sde` – DPM++ 2M SDE scheduler.

The official engine ignores this setting; the control is disabled when using that path.
