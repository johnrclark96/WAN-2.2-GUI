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

# Always echo the invocation so CI can assert on it
Write-Host "[WAN shim] Launch: $PSCommandPath $args"

$python = "D:\\wan22\\venv\\Scripts\\python.exe"
$engine = Join-Path $PSScriptRoot "wan_ps1_engine.py"

if (-not (Test-Path $python)) { throw "Python not found: $python" }
if (-not (Test-Path $engine)) { throw "Engine not found: $engine" }

$argList = @()
if ($mode)        { $argList += @("--mode", $mode) }
if ($prompt)      { $argList += @("--prompt", $prompt) }
if ($neg_prompt)  { $argList += @("--neg_prompt", $neg_prompt) }
if ($sampler)     { $argList += @("--sampler", $sampler) }
if ($steps)       { $argList += @("--steps", $steps) }
if ($cfg)         { $argList += @("--cfg", $cfg) }
if ($seed -ge 0)  { $argList += @("--seed", $seed) }
if ($fps)         { $argList += @("--fps", $fps) }
if ($frames)      { $argList += @("--frames", $frames) }
if ($width)       { $argList += @("--width", $width) }
if ($height)      { $argList += @("--height", $height) }
if ($batch_count) { $argList += @("--batch_count", $batch_count) }
if ($batch_size)  { $argList += @("--batch_size", $batch_size) }
if ($outdir)      { $argList += @("--outdir", $outdir) }
if ($model_dir)   { $argList += @("--model_dir", $model_dir) }
if ($dtype)       { $argList += @("--dtype", $dtype) }
if ($attn)        { $argList += @("--attn", $attn) }
if ($image)       { $argList += @("--image", $image) }
if ($dry_run)     { $argList += "--dry-run" }

& $python $engine @argList
exit $LASTEXITCODE
