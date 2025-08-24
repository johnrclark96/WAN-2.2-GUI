# Environment

The project targets the following wheels and versions:

| Package | Version |
| ------- | ------- |
| `torch` | `2.4.1+cu124` |
| `torchvision` | `0.19.1+cu124` |
| `torchaudio` | `2.4.1+cu124` |
| `diffusers` | `0.35.1` |
| `transformers` | `4.49.0` |
| `tokenizers` | `0.21.x` |
| `accelerate` | `1.1.1` |
| `numpy` | `<2` |

### Flash Attention (Windows, Python 3.10, CUDA 12.4)

```powershell
pip install flash-attn --no-build-isolation --index-url https://flash.attn.wheels/cu124/torch2.4.1
```

Verification snippet:

```python
import torch
import flash_attn  # noqa: F401

print(torch.cuda.is_available())
```

## WAN 2.2 Virtual Environment (Windows)

The engine expects a dedicated virtual environment and directory layout
under `D:\wan22`.  Only the **Wan2.2-TI2V-5B-Diffusers** model is supported.

```powershell
cd D:\wan22
if (!(Test-Path .\venv)) { python -m venv venv }
\venv\Scripts\Activate.ps1

python -m pip install -U pip wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision torchaudio
pip install "diffusers>=0.35.0" transformers>=4.44 accelerate>=0.34 \
    safetensors einops omegaconf imageio imageio-ffmpeg pillow

# dry run
python .\wan_ps1_engine.py --dry-run --mode t2v --prompt "ok" \
    --frames 8 --width 512 --height 288 --attn auto --dtype bfloat16

# tiny real examples
python .\wan_ps1_engine.py --mode t2v --prompt "sunrise" \
    --steps 12 --cfg 6.5 --fps 12 --frames 16 --width 768 --height 432 \
    --outdir D:\wan22\outputs \
    --model_dir D:\wan22\models\Wan2.2-TI2V-5B-Diffusers \
    --dtype bfloat16 --attn auto

python .\wan_ps1_engine.py --mode t2i --prompt "a cat" \
    --steps 8 --width 512 --height 512 \
    --outdir D:\wan22\outputs \
    --model_dir D:\wan22\models\Wan2.2-TI2V-5B-Diffusers \
    --dtype bfloat16 --attn auto
```

