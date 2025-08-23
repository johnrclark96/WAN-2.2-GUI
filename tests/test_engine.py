import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
import types

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


def test_kwargs_whitelist(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(engine, "torch", types.SimpleNamespace())
    utils = types.ModuleType("diffusers.utils")
    utils.export_to_video = lambda *a, **k: None
    diffusers = types.ModuleType("diffusers")
    diffusers.utils = utils
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)

    class DummyImage:
        @staticmethod
        def fromarray(_):
            class Img:
                def save(self, path):
                    Path(path).write_bytes(b"ok")

            return Img()

    class DummyPipe:
        def __call__(self, **kw):  # capture kwargs
            captured.update(kw)
            class R:
                images = [[[0]]]

            return R()

    params = argparse.Namespace(
        prompt="p",
        neg_prompt="",
        height=32,
        width=32,
        frames=1,
        steps=2,
        cfg=1.0,
        batch_size=1,
        batch_count=1,
        fps=8,
        seed=-1,
        mode="t2v",
        image="",
        outdir=str(tmp_path),
    )
    monkeypatch.setattr(engine, "Image", DummyImage)
    engine.run_generation(DummyPipe(), params, "flash", nullcontext())
    assert set(captured.keys()) == {
        "prompt",
        "negative_prompt",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "num_videos_per_prompt",
        "generator",
        "output_type",
    }


def test_png_single_frame(tmp_path, monkeypatch):
    class DummyImage:
        @staticmethod
        def fromarray(_):
            class Img:
                def save(self, path):
                    Path(path).write_bytes(b"ok")

            return Img()

    class DummyPipe:
        def __call__(self, **_):
            class R:
                images = [[[0]]]

            return R()

    params = argparse.Namespace(
        prompt="p",
        neg_prompt="",
        height=32,
        width=32,
        frames=1,
        steps=2,
        cfg=1.0,
        batch_size=1,
        batch_count=1,
        fps=8,
        seed=-1,
        mode="t2v",
        image="",
        outdir=str(tmp_path),
    )
    monkeypatch.setattr(engine, "Image", DummyImage)
    monkeypatch.setattr(engine, "torch", types.SimpleNamespace())
    utils = types.ModuleType("diffusers.utils")
    utils.export_to_video = lambda *a, **k: None
    diffusers = types.ModuleType("diffusers")
    diffusers.utils = utils
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    outputs = engine.run_generation(DummyPipe(), params, "flash", nullcontext())
    assert outputs[0].endswith(".png")
    assert Path(outputs[0]).exists()


def test_mp4_multi_frame(tmp_path, monkeypatch):
    called = {}

    def fake_export(arr, output_video_path, fps):
        called["path"] = output_video_path
        called["fps"] = fps

    class DummyPipe:
        def __call__(self, **_):
            class R:
                frames = [[0, 0]]

            return R()

    params = argparse.Namespace(
        prompt="p",
        neg_prompt="",
        height=32,
        width=32,
        frames=2,
        steps=2,
        cfg=1.0,
        batch_size=1,
        batch_count=1,
        fps=9,
        seed=-1,
        mode="t2v",
        image="",
        outdir=str(tmp_path),
    )
    utils = types.ModuleType("diffusers.utils")
    utils.export_to_video = fake_export
    diffusers = types.ModuleType("diffusers")
    diffusers.utils = utils
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    monkeypatch.setattr(engine, "torch", types.SimpleNamespace())
    outputs = engine.run_generation(DummyPipe(), params, "flash", nullcontext())
    assert outputs[0].endswith(".mp4")
    assert called["fps"] == 9


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
    assert "torch.backends.cuda.sdp_kernel" not in capsys.readouterr().out
