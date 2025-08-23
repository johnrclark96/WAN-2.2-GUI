import shutil
import subprocess
import pytest


def test_runner_path():
    if shutil.which("pwsh") is None:
        pytest.skip("PowerShell not available")
    cmd = ["pwsh", "-NoLogo", "-Command", "./wan_runner.ps1 -dry_run -prompt ok"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert "[WAN shim] Launch:" in result.stdout
    assert "D:\\wan22\\venv\\Scripts\\python.exe" in result.stdout
