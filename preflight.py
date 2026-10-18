"""Verify the reviewed deployment files; does not execute inference."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError(f"Missing or unsafe deployment file: {relative}")
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Wrong file size (possibly LFS pointer): {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {relative}")
    print(f"Verified {len(manifest)} source/model files", flush=True)

if __name__ == "__main__":
    verify()
