# D:\wan22\wan_runner.ps1
# Windows PowerShell 5.1â€“compatible WAN runner
# - Parses --mode/--prompt/... and repeated --lora path:weight
# - Uses venv python if present, else falls back to system python
# - Pipes a here-string into "python -" (PowerShell-safe)
# - Prints "saved: <path>" when the video is written so the UI can pick it up

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = Join-Path $ScriptDir 'venv\Scripts\python.exe'
if (-not (Test-Path $PY)) { $PY = 'python' }

# ---- parse args into hashtable ----
$opts = @{
  mode        = 't2v'
  prompt      = ''
  neg_prompt  = ''
  init_image  = ''
  steps       = 20
  cfg         = 7.0
  seed        = -1
  fps         = 24
  frames      = 48
  width       = 1024
  height      = 576
  sampler     = ''
  batch_count = 1
  batch_size  = 1
  outdir      = (Join-Path $ScriptDir 'outputs')
  model_dir   = (Join-Path $ScriptDir 'models')
  base        = ''
  lora        = @()
  extra       = @()
}

for ($i = 0; $i -lt $args.Count; $i++) {
  $a = [string]$args[$i]
  if ($a -like '--*') {
    $key = $a.Substring(2)
    if ($key -eq 'lora') {
      $i++
      if ($i -lt $args.Count) { $opts.lora += [string]$args[$i] }
      continue
    }
    $i++
    if ($i -lt $args.Count) {
      $val = [string]$args[$i]
      if ($opts.ContainsKey($key)) { $opts[$key] = $val } else { $opts.extra += $val }
    }
  } else {
    $opts.extra += $a
  }
}

# ---- human-friendly log (no ?: ternary) ----
switch ($opts.mode) {
  't2v'  { Write-Host 'Mode: Text->Video' }
  'i2v'  { Write-Host 'Mode: Image->Video' }
  'ti2v' { Write-Host 'Mode: Text+Image->Video' }
  default { Write-Host ("Mode: {0}" -f $opts.mode) }
}

# Ensure output dir exists
if (-not (Test-Path $opts.outdir)) { New-Item -ItemType Directory -Force -Path $opts.outdir | Out-Null }

# Handy dry-run: set $env:WAN_DRYRUN=1 to just echo parsed options and exit 0
if ($env:WAN_DRYRUN -eq '1') {
  Write-Host "DRYRUN opts:" ($opts | ConvertTo-Json -Depth 8)
  exit 0
}

# ---- pass options to Python via env ----
$env:WAN_OPTS_JSON = ($opts | ConvertTo-Json -Depth 8 -Compress)

$py = @'
import os, json, time, random, re, sys
from pathlib import Path

def log(x): print(x, flush=True)

cfg = json.loads(os.environ.get("WAN_OPTS_JSON", "{}"))
mode = cfg.get("mode","t2v")
prompt = cfg.get("prompt","")
neg = cfg.get("neg_prompt","")
init_img = cfg.get("init_image","")
steps = int(cfg.get("steps",20))
cfg_scale = float(cfg.get("cfg",7.0))
seed = int(cfg.get("seed",-1))
fps = int(cfg.get("fps",24))
frames = int(cfg.get("frames",48))
width = int(cfg.get("width",1024))
height = int(cfg.get("height",576))
outdir = Path(cfg.get("outdir","."))
model_dir = cfg.get("base") or cfg.get("model_dir") or "models"
loras = cfg.get("lora",[])

try:
    import torch
    from diffusers import DiffusionPipeline, AutoPipelineForText2Image
except Exception as e:
    log(f"[ps1] Diffusers not available: {e}")
    sys.exit(2)

pipe = None
try:
    log(f"[ps1] Loading: {model_dir}")
    pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
except Exception as e:
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=torch.float16)
    except Exception as e2:
        log(f"[ps1] Could not load pipeline: {e} / {e2}")
        sys.exit(3)

if hasattr(pipe,"enable_model_cpu_offload"):
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# ---- LoRA apply (best-effort) ----
if loras:
    try:
        from safetensors.torch import load_file as safe_load
        import torch as _torch
        tgt = getattr(pipe,"unet",None) or getattr(pipe,"transformer",None)
        if tgt is not None:
            base_sd = {k:(v.float().clone() if hasattr(v,"is_floating_point") and v.is_floating_point() else v.clone())
                       for k,v in tgt.state_dict().items()}
            def groups(sd):
                for k in list(sd.keys()):
                    if k.endswith("lora_up.weight"):
                        base = k[:-len("lora_up.weight")]
                        down = base + "lora_down.weight"
                        alpha = base + "alpha"
                        if down in sd:
                            yield base, k, down, (alpha if alpha in sd else None)
            for item in loras:
                path, _, weight = item.partition(":")
                try: scale = float(weight) if weight else 0.8
                except: scale = 0.8
                p = Path(path)
                if not p.exists():
                    log(f"[ps1] LoRA missing: {p}")
                    continue
                sd = safe_load(str(p)) if p.suffix.lower()==".safetensors" else _torch.load(str(p), map_location="cpu")
                for base, upk, downk, alphak in groups(sd):
                    tk = base + "weight"
                    if tk not in base_sd:
                        if tk.endswith(".") and tk[:-1] in base_sd:
                            tk = tk[:-1]
                        else:
                            continue
                    up = sd[upk].float(); down = sd[downk].float()
                    r = down.shape[0]; alpha = sd[alphak].item() if (alphak and alphak in sd) else r
                    delta = (up @ down) * (alpha / r) * scale
                    if hasattr(base_sd[tk],"shape") and base_sd[tk].shape != delta.shape:
                        continue
                    base_sd[tk].add_(delta)
            mdtype = next(tgt.parameters()).dtype
            for k,v in list(base_sd.items()):
                if hasattr(v,"is_floating_point") and v.is_floating_point():
                    base_sd[k] = v.to(mdtype)
            tgt.load_state_dict(base_sd, strict=False)
    except Exception as e:
        log(f"[ps1] LoRA apply failed: {e}")

# ---- generation ----
if seed is None or seed < 0:
    seed = random.randint(1, 2**31-1)
g = None
try:
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
except Exception:
    pass

common = dict(prompt=prompt, negative_prompt=neg, num_inference_steps=steps,
              guidance_scale=cfg_scale, width=width, height=height)
if g is not None: common["generator"] = g
if mode in ("i2v","ti2v") and init_img: common["image"] = init_img

variants = [
    {"num_frames": frames, "fps": fps},
    {"video_length": frames, "fps": fps},
    {"num_frames": frames},
    {"video_length": frames},
    {}
]

out = None; last = None
for v in variants:
    try:
        kw = dict(common); kw.update(v)
        out = pipe(**kw); break
    except TypeError as e:
        last = e; continue
if out is None:
    log(f"[ps1] Pipeline call failed: {last}")
    sys.exit(4)

outdir.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
vp = outdir / f"WAN22_{mode}_{ts}.mp4"

def save_frames(frames):
    try:
        import numpy as np, imageio
        arr = [np.array(f) for f in frames]
        imageio.mimwrite(str(vp), arr, fps=max(1, fps))
        log(f"saved: {vp}")
        return True
    except Exception as e:
        log(f"[ps1] save error: {e}"); return False

ok = False
try:
    if hasattr(out,"frames"):
        ok = save_frames(out.frames)
    elif isinstance(out,dict) and "frames" in out:
        ok = save_frames(out["frames"])
except Exception:
    pass

if not ok:
    try:
        vid = getattr(out,"videos",None) if not isinstance(out,dict) else out.get("videos")
        if vid is not None:
            import numpy as np, imageio
            v = vid[0] if getattr(vid,"ndim",0)==5 else vid
            v = (v.permute(0,2,3,1).clamp(0,1).cpu().numpy()*255).astype("uint8")
            imageio.mimwrite(str(vp), v, fps=max(1, fps))
            log(f"saved: {vp}")
            ok = True
    except Exception as e:
        log(f"[ps1] tensor save error: {e}")

if not ok:
    sp = outdir / f"WAN22_{mode}_{ts}.json"
    with open(sp,"w",encoding="utf-8") as f:
        json.dump({"status":"ok","note":"No video detected"}, f)
    log(f"wrote: {sp}")
