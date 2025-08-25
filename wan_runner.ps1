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
    [string]$dtype = "bfloat16",
    [string]$attn = "auto",
    [string]$image,
    [switch]$dry_run
)

$ErrorActionPreference = "Stop"

$python = "D:\\wan22\\venv\\Scripts\\python.exe"
$engine = Join-Path $PSScriptRoot "wan_ps1_engine.py"

$env:HF_HOME = "D:\\wan22.cache\\huggingface"
$env:TRANSFORMERS_CACHE = "D:\\wan22.cache\\huggingface\\hub"
$env:PYTHONIOENCODING = "utf-8"

foreach ($d in @("D:\\wan22\\outputs", "D:\\wan22\\json")) {
  if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

if (-not (Test-Path $python)) {
  Write-Host "[WAN shim] Launch: $python (missing)"
  Write-Error "WAN venv python not found at $python"
  exit 1
}
if (-not (Test-Path $engine)) { throw "Engine not found: $engine" }

$argv = @($engine)
if ($mode)        { $argv += @("--mode", $mode) }
if ($prompt)      { $argv += @("--prompt", ($prompt -replace '"', '\"')) }
if ($neg_prompt)  { $argv += @("--neg_prompt", ($neg_prompt -replace '"', '\"')) }
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
if ($image)       { $argv += @("--image", $image) }
if ($dry_run)     { $argv += "--dry-run" }

Write-Host "[WAN shim] Launch: $python $($argv -join ' ')"
& $python @argv
exit $LASTEXITCODE
