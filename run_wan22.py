"""Thin Python runner for WAN 2.2.

This replaces the previous PowerShell-based launcher.  It forwards all
arguments to ``wan_ps1_engine.py`` and mirrors its output to stdout while
also providing a very small console progress indicator for direct CLI
usage.  Environment variables previously configured in ``wan_runner.ps1``
are also handled here so the UI no longer relies on a PowerShell layer.
"""

import os, re, sys, subprocess
from pathlib import Path


def main() -> int:
    here = Path(__file__).parent
    engine = here / "wan_ps1_engine.py"
    if not engine.exists():
        print(f"[ERROR] Missing engine: {engine}")
        return 1

    # Forward all CLI args straight into the engine
    cmd = [sys.executable, "-u", str(engine)] + sys.argv[1:]

    # Pretty-print the launched command for easier debugging/path detection
    shown = " ".join(
        f'"{c}"' if (" " in c and not c.startswith("-")) else c for c in cmd
    )
    print(f"[WAN runner] Launch: {shown}")

    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256,garbage_collection_threshold:0.9",
    }

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(here),
            env=env,
        )
    except (OSError, FileNotFoundError) as e:
        print(f"[ERROR] Failed to launch engine: {e}")
        return 1

    prog_re = re.compile(
        r"^\[PROGRESS\]\s+step=(\d+)/(\d+)\s+frame=(\d+)/(\d+)\s+percent=(\d+)"
    )
    last_pct = -1

    assert proc.stdout is not None
    for line in proc.stdout:
        m = prog_re.search(line)
        if m:
            step, steps, frame, frames, pct = map(int, m.groups())
            if pct != last_pct:
                status = f"Frame {frame}/{frames} · Step {step}/{steps}"
                sys.stderr.write(f"\r{status} {pct}%")
                sys.stderr.flush()
                last_pct = pct
        else:
            print(line, end="")

    proc.wait()
    if last_pct >= 0:
        sys.stderr.write("\n")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
