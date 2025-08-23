import argparse
import sys
from pathlib import Path
from contextlib import nullcontext
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wan_ps1_engine as engine  # noqa: E402


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
