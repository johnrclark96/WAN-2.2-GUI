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

