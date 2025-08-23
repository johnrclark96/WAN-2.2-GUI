import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wan_ps1_engine import main


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "WAN"
    (model_dir / "vae").mkdir(parents=True)
    cfg = {
        "base_dim": 160,
        "z_dim": 48,
        "scale_factor_spatial": 1,
        "scale_factor_temporal": 1,
    }
    (model_dir / "vae" / "config.json").write_text(json.dumps(cfg))
    return model_dir


def test_dry_run_backend(monkeypatch, tmp_path, capsys):
    model_dir = make_model_dir(tmp_path)
    argv = [
        "wan_ps1_engine.py",
        "--dry-run",
        "--attn",
        "auto",
        "--mode",
        "t2v",
        "--frames",
        "8",
        "--width",
        "512",
        "--height",
        "288",
        "--model_dir",
        str(model_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    dummy = mock.Mock()
    with mock.patch.dict(sys.modules, {"diffusers": dummy}):
        with pytest.raises(SystemExit) as e:
            main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Attention backend" in out
    dummy.DiffusionPipeline.from_pretrained.assert_not_called()
