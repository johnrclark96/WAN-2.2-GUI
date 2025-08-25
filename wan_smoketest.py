"""WAN smoketest helper.

Validates command building for both engines without loading models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.paths import OFFICIAL_GENERATE, OUTPUT_DIR, VENV_PY


def build_cmd(engine: str) -> list[str]:
    if engine == "diffusers":
        return ["pwsh", "-NoLogo", "-File", "wan_runner.ps1"]
    if engine == "official":
        if not OFFICIAL_GENERATE:
            raise ValueError("OFFICIAL_GENERATE unset")
        return [str(VENV_PY), OFFICIAL_GENERATE]
    raise ValueError(f"unknown engine {engine}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["diffusers", "official"], default="diffusers")
    ap.add_argument("--mode", choices=["build-only"], default="build-only")
    args = ap.parse_args()
    cmd = build_cmd(args.engine)
    print(json.dumps({"cmd": cmd, "outdir": Path(OUTPUT_DIR).as_posix()}))


if __name__ == "__main__":
    main()
