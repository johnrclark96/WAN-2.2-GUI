import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wan_ps1_engine as engine  # noqa: E402


def test_attention_api(tmp_path, monkeypatch, capsys):
    class DummyPipe:
        def to(self, _):
            pass

        def enable_model_cpu_offload(self):
            pass

        vae = type("V", (), {"to": staticmethod(lambda _dtype: None)})()
        scheduler = type("S", (), {"config": {}})()
        transformer = type("T", (), {"to": staticmethod(lambda _dev: None)})()

    monkeypatch.setattr(engine, "load_pipeline", lambda *a: DummyPipe())
    monkeypatch.setattr(engine, "run_generation", lambda *a: [])
    argv = [
        "wan_ps1_engine.py",
        "--mode",
        "t2v",
        "--prompt",
        "hi",
        "--steps",
        "1",
        "--frames",
        "1",
        "--width",
        "32",
        "--height",
        "32",
        "--outdir",
        str(tmp_path),
        "--model_dir",
        "m",
        "--attn",
        "auto",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    code = engine.main()
    assert code == 0
    out = capsys.readouterr().out
    deprecated = "torch.backends.cuda." + "sdp_kernel"
    assert deprecated not in out
