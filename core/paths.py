"""Centralized paths for WAN 2.2 GUI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(r"D:\\wan22\\wan_paths.json")

_DEF_ROOT = Path(r"D:\\wan22")
_DEF_PY_EXE = _DEF_ROOT / "venv" / "Scripts" / "python.exe"
_DEF_PS1 = _DEF_ROOT / "wan_runner.ps1"
_DEF_OFFICIAL = _DEF_ROOT / "generate.py"
_DEF_OUTPUT = _DEF_ROOT / "outputs"
_DEF_JSON = _DEF_ROOT / "json"
_DEF_MODELS = _DEF_ROOT / "models"
_DEF_CKPT = _DEF_MODELS / "Wan2.2-TI2V-5B"

_defaults: Dict[str, Any] = {
    "WAN22_ROOT": _DEF_ROOT,
    "PY_EXE": _DEF_PY_EXE,
    "PS1_ENGINE": _DEF_PS1,
    "OFFICIAL_GENERATE": _DEF_OFFICIAL,
    "OUTPUT_DIR": _DEF_OUTPUT,
    "JSON_DIR": _DEF_JSON,
    "MODELS_DIR": _DEF_MODELS,
    "CKPT_TI2V_5B": _DEF_CKPT,
}

_config: Dict[str, Any] = {}
try:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        _config = json.load(fh)
except (FileNotFoundError, json.JSONDecodeError):
    _config = {}


def _resolve(key: str) -> Any:
    env_val = os.getenv(key)
    if env_val is not None:
        return env_val
    if key in _config:
        return _config[key]
    return _defaults[key]


WAN22_ROOT = Path(_resolve("WAN22_ROOT"))
PY_EXE = Path(_resolve("PY_EXE"))
PS1_ENGINE = Path(_resolve("PS1_ENGINE"))
OFFICIAL_GENERATE = Path(_resolve("OFFICIAL_GENERATE"))
OUTPUT_DIR = Path(_resolve("OUTPUT_DIR"))
JSON_DIR = Path(_resolve("JSON_DIR"))
MODELS_DIR = Path(_resolve("MODELS_DIR"))
CKPT_TI2V_5B = Path(_resolve("CKPT_TI2V_5B"))

# Backward compatibility
VENV_PY = PY_EXE


def save_config(update: Dict[str, Any]) -> None:
    """Persist path overrides back to ``wan_paths.json``."""
    cfg = _config.copy()
    cfg.update(update)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

__all__ = [
    "WAN22_ROOT",
    "PY_EXE",
    "PS1_ENGINE",
    "OUTPUT_DIR",
    "JSON_DIR",
    "MODELS_DIR",
    "CKPT_TI2V_5B",
    "OFFICIAL_GENERATE",
    # compatibility
    "VENV_PY",
]
