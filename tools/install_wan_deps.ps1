# Install WAN 2.2 dependencies into D:\wan22\venv
cd D:\wan22
if (!(Test-Path .\venv)) { python -m venv venv }
\venv\Scripts\Activate.ps1

# Ensure pip latest
python -m pip install -U pip wheel

# Core CUDA 12.1 builds
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.* torchvision torchaudio

# Core libraries
pip install diffusers==0.35.* accelerate==0.34.* transformers==4.44.* \
    safetensors einops omegaconf imageio imageio-ffmpeg

# Optional FlashAttention (Hopper only)
try {
    $cap = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader) 2>$null
    if ($cap -ge 9.0) {
        pip install flash-attn --no-build-isolation --index-url https://flash.attn.wheels/cu121/torch2.4.0
    }
} catch {
    Write-Host "Skipping flash-attn install"
}

Write-Host "`n[OK] Dependencies installed."
