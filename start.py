import os, sys
from pathlib import Path
from preflight import verify
os.chdir(Path(__file__).resolve().parent)
verify()
port = int(os.environ.get("PORT", "7860"))
if not 1 <= port <= 65535:
    raise ValueError("Invalid PORT")
os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "Streamlit/rwcdx.py",
    "--server.address=0.0.0.0", f"--server.port={port}", "--server.headless=true",
    "--browser.gatherUsageStats=false", "--server.fileWatcherType=none",
    "--server.enableStaticServing=false"])
