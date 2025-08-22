# D:\wan22\run_wan22.py
import sys, subprocess
from pathlib import Path

def main():
    here = Path(__file__).parent
    ps1 = here / "wan_runner.ps1"
    if not ps1.exists():
        print(f"[ERROR] Missing runner: {ps1}")
        sys.exit(1)

    # Forward all CLI args from the UI straight into the PS1
    cmd = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(ps1)
    ] + sys.argv[1:]

    # Pretty print the launched command (helps the UI detect paths)
    shown = " ".join(f'"{c}"' if (" " in c and not c.startswith("-")) else c for c in cmd)
    print(f"[WAN shim] Launch: {shown}")

    # Stream logs to stdout so the web UI console shows them live
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(here),
        )
    except (OSError, FileNotFoundError) as e:
        print(f"[ERROR] Failed to launch PowerShell or runner: {e}")
        sys.exit(1)

    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
