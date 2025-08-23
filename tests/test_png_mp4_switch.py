import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wan_ps1_engine as engine  # noqa: E402


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
