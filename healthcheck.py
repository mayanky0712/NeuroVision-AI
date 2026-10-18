import os
from urllib.request import urlopen
port = int(os.environ.get("PORT", "7860"))
with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=4) as response:
    if response.status != 200 or response.read().strip() != b"ok":
        raise SystemExit(1)
