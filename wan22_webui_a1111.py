#!/usr/bin/env python
# wan22_webui_a1111.py – A1111-style Gradio UI for WAN 2.2

import argparse
import socket
from gui.layout import build_app

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--listen", action="store_true", help="Allow other PCs to access, bind to 0.0.0.0")
    ap.add_argument("--auth", type=str, default=None, help="Optional basic auth (user:pass)")
    args = ap.parse_args()

    ui = build_app()
    auth = tuple(args.auth.split(":", 1)) if args.auth and ":" in args.auth else None

    # Auto-switch port if 7860 is busy (e.g., if Stable Diffusion WebUI is running)
    chosen_port = args.port
    if args.port == 7860 and is_port_in_use(7860):
        if not is_port_in_use(7861):
            print("[ui] Port 7860 busy; switching to 7861.", flush=True)
            chosen_port = 7861
        else:
            print("[ui] Ports 7860 and 7861 are busy. Using 7860 and letting Gradio handle the conflict.", flush=True)

    ui.launch(
        server_name=("0.0.0.0" if args.listen else "127.0.0.1"),
        server_port=chosen_port,
        auth=auth,
        inbrowser=False,
        show_api=False,
    )

if __name__ == "__main__":
    main()
