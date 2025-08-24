import sys
import shutil
import subprocess
import pytest
import pathlib


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell runner is Windows-only")
def test_runner_path():
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        pytest.skip("No PowerShell available")

    cmd = [
        exe,
        "-NoLogo",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "wan_runner.ps1",
        "-dry_run",
        "-prompt",
        "ok",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(pathlib.Path.cwd()),
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0
    assert "[WAN shim] Launch:" in result.stdout

