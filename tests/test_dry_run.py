import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wan_ps1_engine import main  # noqa: E402


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


@pytest.mark.parametrize(
    "mesh,compile_mode",
    [("off", "off"), ("grid", "reduce-overhead")],
)
def test_dry_run_backend(monkeypatch, tmp_path, capsys, mesh, compile_mode):
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
    if mesh != "off":
        argv += ["--mesh", mesh]
    if compile_mode != "off":
        argv += ["--compile", compile_mode]
    monkeypatch.setattr(sys, "argv", argv)
    dummy = mock.Mock()
    with mock.patch.dict(sys.modules, {"diffusers": dummy}):
        with pytest.raises(SystemExit) as e:
            main()
    assert e.value.code == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert out.startswith("[RESULT] OK DRY_RUN ")
    data = json.loads(out.split("DRY_RUN ", 1)[1])
    assert data["mesh"] == mesh
    assert data["compile"] == compile_mode
    dummy.DiffusionPipeline.from_pretrained.assert_not_called()
