import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

gui = pytest.importorskip("wan22_webui_a1111")  # noqa: E402


def test_model_dir_exists_check(tmp_path):
    gen = gui.run_cmd(
        engine="diffusers",
        mode="t2v",
        prompt="ok",
        neg_prompt="",
        sampler="unipc",
        steps=1,
        cfg=1.0,
        seed=-1,
        fps=8,
        frames=1,
        width=32,
        height=32,
        batch_count=1,
        batch_size=1,
        outdir=str(tmp_path),
        model_dir=str(tmp_path / "missing"),
        dtype="bf16",
        attn="sdpa",
        image=None,
    )
    with pytest.raises(gui.gr.Error):
        next(gen)
