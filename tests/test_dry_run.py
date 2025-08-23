import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wan_ps1_engine as engine  # noqa: E402


def run_main(monkeypatch, tmp_path, args):
    argv = ["wan_ps1_engine.py"] + args + ["--outdir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    return engine.main()


def read_last_line(capsys):
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_dry_run_ok(tmp_path, monkeypatch, capsys):
    code = run_main(
        monkeypatch,
        tmp_path,
        [
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
        ],
    )
    assert code == 0
    out_line = read_last_line(capsys)
    assert out_line.startswith("[RESULT] OK ")
    data = json.loads(out_line.split("[RESULT] OK ", 1)[1])
    assert data["config"]["frames"] == 8
    assert list(tmp_path.glob("dryrun_*.json"))


@pytest.mark.parametrize(
    "extra",
    [
        ["--prompt", "x", "--frames", "0"],
        ["--prompt", "x", "--steps", "0"],
        [],
        ["--prompt", "x", "--mode", "i2v"],
    ],
)
def test_invalid_inputs_blocked(tmp_path, monkeypatch, capsys, extra):
    code = run_main(monkeypatch, tmp_path, ["--dry-run"] + extra)
    assert code == 1
    assert read_last_line(capsys).startswith("[RESULT] FAIL GENERATION")
