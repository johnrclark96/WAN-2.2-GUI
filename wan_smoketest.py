"""WAN smoketest helper.

Validates command building for both engines without loading models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.paths import CKPT_TI2V_5B, OFFICIAL_GENERATE, OUTPUT_DIR, VENV_PY


def build_cmd(engine: str, width: int = 1280, height: int = 704, free_gb: float = 32.0) -> list[str]:
    if engine == "diffusers":
        return ["pwsh", "-NoLogo", "-File", "wan_runner.ps1"]
    if engine == "official":
        if not OFFICIAL_GENERATE:
            raise ValueError("OFFICIAL_GENERATE unset")
        w = width // 32 * 32
        h = height // 32 * 32
        if h != 704:
            raise ValueError("official height must be 704")
        cmd = [
            str(VENV_PY),
            OFFICIAL_GENERATE,
            "--task",
            "ti2v-5B",
            "--prompt",
            "test",
            "--ckpt_dir",
            str(CKPT_TI2V_5B),
            "--size",
            f"{w}*{h}",
        ]
        if free_gb < 24:
            cmd += ["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"]
        return cmd
    raise ValueError(f"unknown engine {engine}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["diffusers", "official"], default="diffusers")
    ap.add_argument("--mode", choices=["build-only"], default="build-only")
    ap.add_argument("--free_gb", type=float, default=32.0)
    args = ap.parse_args()
    cmd = build_cmd(args.engine, free_gb=args.free_gb)
    print(json.dumps({"cmd": cmd, "outdir": Path(OUTPUT_DIR).as_posix()}))


if __name__ == "__main__":
    main()
