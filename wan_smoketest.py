import sys
from pathlib import Path
from diffusers import DiffusionPipeline

base = r"D:\wan22\models\Wan2.2-TI2V-5B-Diffusers"
if not (Path(base)/"model_index.json").exists():
    print("MISSING model_index.json at", base); raise SystemExit(3)

pipe = DiffusionPipeline.from_pretrained(base, trust_remote_code=True, torch_dtype="auto")
print("LOAD_OK")
