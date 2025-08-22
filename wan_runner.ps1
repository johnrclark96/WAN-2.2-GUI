$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:true,max_split_size_mb:128,garbage_collection_threshold:0.8"
# wan_runner.ps1 — progress-aware runner that shows a console progress bar
$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "venv\Scripts\python.exe"
$engine = Join-Path $root "wan_ps1_engine.py"

if (-not (Test-Path $python)) { Write-Error "Python not found: $python" }
if (-not (Test-Path $engine)) { Write-Error "Engine not found: $engine" }

# Build argument line (quote each arg)
$argv = @()
foreach ($a in $args) {
  if ($a -match '\s' -or $a -match '["'']') { $argv += ('"'+$a.replace('"','`"')+'"') }
  else { $argv += $a }
}
$argline = ('"'+$engine+'" ' + ($argv -join ' '))

# Start python with redirected IO so we can parse progress lines
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $python
$psi.Arguments = $argline
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.CreateNoWindow = $true

$proc = [System.Diagnostics.Process]::Start($psi)

# Read output line-by-line and update progress
$activity = "WAN 2.2 generation"
$lastPercent = -1
while (-not $proc.HasExited) {
  $line = $proc.StandardOutput.ReadLine()
  if ($null -ne $line) {
    Write-Host $line
    if ($line -match '^\[PROGRESS\]\s+step=(\d+)/(\d+)\s+frame=(\d+)/(\d+)\s+percent=(\d+)') {
      $step = [int]$matches[1]; $steps=[int]$matches[2]
      $frame=[int]$matches[3];  $frames=[int]$matches[4]
      $pct  = [int]$matches[5]
      if ($pct -ne $lastPercent) {
        $status = "Frame $frame/$frames · Step $step/$steps"
        Write-Progress -Id 1 -Activity $activity -Status $status -PercentComplete $pct
        $lastPercent = $pct
      }
    }
  } else {
    Start-Sleep -Milliseconds 50
  }
}

# Drain remaining output
while (-not $proc.StandardOutput.EndOfStream) {
  $line = $proc.StandardOutput.ReadLine()
  if ($line) { Write-Host $line }
}
while (-not $proc.StandardError.EndOfStream) {
  $line = $proc.StandardError.ReadLine()
  if ($line) { Write-Warning $line }
}

Write-Progress -Id 1 -Activity $activity -Completed
exit $proc.ExitCode

