import sys
import shutil
import subprocess
import pytest


def test_runner_path():
    if sys.platform != "win32":
        pytest.skip("PowerShell runner only works on Windows")
    if shutil.which("pwsh") is None and shutil.which("powershell") is None:
        pytest.skip("No PowerShell available")
    cmd = ["pwsh", "-NoLogo", "-Command", "./wan_runner.ps1 -dry_run -prompt ok"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert "[WAN shim] Launch:" in result.stdout

