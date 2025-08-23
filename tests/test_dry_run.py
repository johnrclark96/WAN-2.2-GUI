import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wan_ps1_engine import main  # noqa: E402


def test_dry_run(tmp_path, monkeypatch, capsys):
    outdir = tmp_path / "out"
    argv = [
        "wan_ps1_engine.py",
        "--dry-run",
        "--mode",
        "t2v",
        "--prompt",
        "ok",
        "--frames",
        "8",
        "--width",
        "512",
        "--height",
        "288",
        "--outdir",
        str(outdir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    code = main()
    assert code == 0
    out_line = capsys.readouterr().out.strip().splitlines()[-1]
    assert out_line.startswith("[RESULT] OK ")
    data = json.loads(out_line.split("[RESULT] OK ", 1)[1])
    assert data["ok"] is True
    assert data["config"]["frames"] == 8
    assert list(outdir.glob("dryrun_*.json"))
