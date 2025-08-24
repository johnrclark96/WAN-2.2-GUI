# WAN 2.2 GUI

Minimal runner and GUI for the WAN 2.2 video generation pipelines. The
project assumes a Windows layout rooted at `D:\wan22`:

```
D:\wan22\venv\            # Python virtual environment
D:\wan22\models\Wan2.2-TI2V-5B-Diffusers\  # 5B model
D:\wan22\outputs\         # Generated media
```

Only the **5B TI2V** diffusers model is supported and used by both the CLI
and GUI.

## First run

```powershell
cd D:\wan22
pwsh -NoLogo -ExecutionPolicy Bypass -File wan_runner.ps1 --help
pwsh -NoLogo -ExecutionPolicy Bypass -File wan_runner.ps1 --dry_run --mode t2v --prompt ok

# tiny real examples
pwsh -NoLogo -ExecutionPolicy Bypass -File wan_runner.ps1 --mode t2v --prompt "sunrise" --steps 8 --frames 16 --width 768 --height 432
pwsh -NoLogo -ExecutionPolicy Bypass -File wan_runner.ps1 --mode t2i --prompt "a cat" --steps 8 --width 512 --height 512
```

## Runtime setup

See [docs/env.md](docs/env.md) for pinned dependency versions and installation
notes.

See [CHANGELOG.md](CHANGELOG.md) for recent updates.
