"""
Local-network launcher for the Raman/SERS Bacterial Spectra Matcher.

Run this instead of app.py directly when hosting on the lab PC for
colleagues to reach over the local network. Same app, same logic —
this just adds password protection and binds to the network interface
instead of localhost-only.

USAGE:
    Set credentials as environment variables before running (recommended,
    keeps the password out of the source file):

        Windows (cmd):      set RAMAN_APP_USER=labuser
                             set RAMAN_APP_PASSWORD=yourpassword
                             python local_app.py

        Windows (PowerShell): $env:RAMAN_APP_USER="labuser"
                               $env:RAMAN_APP_PASSWORD="yourpassword"
                               python local_app.py

    Or just double-click start_server.bat, which sets these for you —
    edit the username/password inside that file.

    If no environment variables are set, falls back to the defaults
    below (change these before handing this out).
"""

import os
import socket
from app import demo

DEFAULT_USER = "labuser"
DEFAULT_PASSWORD = "changeme123"

USERNAME = os.environ.get("RAMAN_APP_USER", DEFAULT_USER)
PASSWORD = os.environ.get("RAMAN_APP_PASSWORD", DEFAULT_PASSWORD)
PORT = int(os.environ.get("RAMAN_APP_PORT", "7860"))


def get_local_ip():
    """Best-effort guess at this machine's LAN IP, for the printed access URL."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    if PASSWORD == DEFAULT_PASSWORD:
        print("\n*** WARNING: using the default password. Set RAMAN_APP_PASSWORD "
              "before sharing this with colleagues. ***\n")

    local_ip = get_local_ip()
    print(f"\nStarting Raman/SERS Matcher on the local network...")
    print(f"  Username: {USERNAME}")
    print(f"  On this PC:        http://localhost:{PORT}")
    print(f"  From other PCs on the same network: http://{local_ip}:{PORT}")
    print(f"  (colleagues need Windows Firewall to allow inbound connections on port {PORT})\n")

    demo.launch(
        server_name="0.0.0.0",   # listen on all network interfaces, not just localhost
        server_port=PORT,
        auth=(USERNAME, PASSWORD),
        auth_message="Enter your lab credentials to access the spectra matcher.",
        share=False,              # no public Gradio link — local network only
    )
