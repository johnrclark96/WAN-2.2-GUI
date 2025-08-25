# Environment

The project targets Windows with CUDA 12.1 and Python 3.10/3.11.  All dependencies live in `D:\wan22\venv`.

| Package | Version |
| ------- | ------- |
| `torch` | `2.4.*` (cu121) |
| `torchvision` | `matching` |
| `torchaudio` | `matching` |
| `diffusers` | `0.35.*` |
| `transformers` | `4.44.*` |
| `accelerate` | `0.34.*` |
| `safetensors` | latest |
| `einops` | latest |
| `omegaconf` | latest |
| `imageio` | latest |
| `imageio-ffmpeg` | latest |

### Flash Attention (Hopper only)

```powershell
# Requires SM >= 90
pip install flash-attn --no-build-isolation --index-url https://flash.attn.wheels/cu121/torch2.4.0
```

## WAN 2.2 Virtual Environment (Windows)

The WAN engine lives in its own virtual environment.  Paths assume `D:\wan22` as the root directory.

```powershell
cd D:\wan22
if (!(Test-Path .\venv)) { python -m venv venv }
\venv\Scripts\Activate.ps1

python -m pip install -U pip wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.* torchvision torchaudio
pip install diffusers==0.35.* accelerate==0.34.* transformers==4.44.* \
    safetensors einops omegaconf imageio imageio-ffmpeg pillow

# smoke test (no models required)
python .\wan_ps1_engine.py --dry-run --mode t2v --prompt "ok" \
    --frames 8 --fps 24 --width 1280 --height 704 --attn auto --dtype bfloat16
```

If a model directory is present you can perform a tiny real test:

```powershell
python .\wan_ps1_engine.py --mode t2v --prompt "sunrise over the ocean" \
    --steps 12 --cfg 6.5 --fps 12 --frames 16 --width 768 --height 432 \
    --outdir D:\wan22\outputs \
    --dtype bfloat16 --attn auto --seed 123
```

### Recommended defaults for 16 GB GPUs

For cards with 16 GB of VRAM these options help avoid out of memory errors:

```
--offload_model true --convert_model_dtype true --t5_cpu true
```
