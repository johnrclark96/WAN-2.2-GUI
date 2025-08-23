# Create & activate venv
cd D:\wan22
if (!(Test-Path .\venv)) { python -m venv venv }
.\venv\Scripts\activate

# CUDA 12.1 build for RTX 4090
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core libs
pip install diffusers==0.31.0 transformers==4.49.0 accelerate safetensors einops omegaconf
# IO & video
pip install imageio imageio-ffmpeg decord

Write-Host "`n[OK] Dependencies installed."
Write-Host "Tip: If MP4 saving fails, install FFmpeg:  winget install Gyan.FFmpeg"
