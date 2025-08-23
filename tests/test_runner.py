import sys
import shutil
import subprocess
import pytest


def test_runner_path():
    if sys.platform != "win32":
        pytest.skip("PowerShell runner only works on Windows")
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if exe is None:
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert "[WAN shim] Launch:" in result.stdout

