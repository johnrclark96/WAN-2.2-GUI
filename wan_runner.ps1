param(
    [string]$mode,
    [string]$prompt,
    [string]$neg_prompt,
    [string]$sampler = "unipc",
    [int]$steps = 20,
    [double]$cfg = 7.0,
    [long]$seed = -1,
    [int]$fps = 24,
    [int]$frames = 16,
    [int]$width = 768,
    [int]$height = 432,
    [int]$batch_count = 1,
    [int]$batch_size = 1,
    [string]$outdir = "D:/wan22/outputs",
    [string]$model_dir = "D:/wan22/models/Wan2.2-TI2V-5B-Diffusers",
    [string]$dtype = "bf16",
    [string]$attn = "sdpa",
    [string]$offload = "none",
    [string]$image,
    [switch]$dry_run
)

$ErrorActionPreference = "Stop"

# Explicit virtual environment interpreter and engine path
$python = "D:\wan22\venv\Scripts\python.exe"
$engine = Join-Path $PSScriptRoot "wan_ps1_engine.py"

# Caches / allocator hygiene
$null = $null  # keep clean, rely on HF_HOME
$env:HF_HOME = "D:\wan22\.cache\huggingface"
$env:PYTHONIOENCODING = "utf-8"

# Calmer allocator + lazy CUDA module loading
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.9"
$env:CUDA_MODULE_LOADING = "LAZY"


# Ensure logs dir exists for persistent run logs
@("D:\wan22\logs") | ForEach-Object { if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null } }

# Ensure required directories exist (pipeline form avoids nested parentheses)
@(
  "D:\wan22\outputs",
  "D:\wan22\json",
  "D:\wan22\.cache\huggingface",
  "D:\wan22\.cache\huggingface\hub"
) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}

# Basic sanity checks
if (-not (Test-Path $python)) {
    Write-Host "[WAN shim] Launch: $python (missing)"
    Write-Error "WAN virtual environment python not found at $python"
    exit 1
}
if (-not (Test-Path $engine)) {
    throw "Engine not found: $engine"
}

# Build argv for the engine
$argv = @($engine)
if ($mode)        { $argv += @("--mode", $mode) }
if ($prompt)      { $argv += @("--prompt", $prompt) }
if ($neg_prompt)  { $argv += @("--neg_prompt", $neg_prompt) }
if ($sampler)     { $argv += @("--sampler", $sampler) }
if ($steps)       { $argv += @("--steps", $steps) }
if ($cfg)         { $argv += @("--cfg", $cfg) }
if ($seed -ge 0)  { $argv += @("--seed", $seed) }
if ($fps)         { $argv += @("--fps", $fps) }
if ($frames)      { $argv += @("--frames", $frames) }
if ($width)       { $argv += @("--width", $width) }
if ($height)      { $argv += @("--height", $height) }
if ($batch_count) { $argv += @("--batch_count", $batch_count) }
if ($batch_size)  { $argv += @("--batch_size", $batch_size) }
if ($outdir)      { $argv += @("--outdir", $outdir) }
if ($model_dir)   { $argv += @("--model_dir", $model_dir) }
if ($dtype)       { $argv += @("--dtype", $dtype) }
if ($attn)        { $argv += @("--attn", $attn) }
if ($offload)     { $argv += @("--offload", $offload) }
if ($image)       { $argv += @("--image", $image) }
if ($dry_run)     { $argv += "--dry-run" }

# Print a simple launch line without $() subexpressions (avoids parse edge cases)
$cmdline = "$python " + ($argv -join ' ')
Write-Host "[WAN shim] Launch: $cmdline"

& $python @argv
exit $LASTEXITCODE
