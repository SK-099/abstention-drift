"""Download the fine-tuned answerability checker from the GitHub release.

The weights (about 1.1 GB) are too large for the git repository, so they are
published as assets of the release "checker-v1". This script downloads them
into checkpoints/checker/best/ and verifies their SHA-256 checksums.
(threshold.json and checker_mode.json are already in the repository.)

    python scripts/get_checker.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

TAG = "checker-v1"
BASE = f"https://github.com/SK-099/abstention-drift/releases/download/{TAG}"
DEST = Path(__file__).resolve().parents[1] / "checkpoints" / "checker" / "best"
FILES = {
    "config.json": "51c1782a4115af553a5396112ae680adeb688cf038ff5d19a8c71da8a96329c8",
    "tokenizer.json": "d2b37bc8cd179e87381964173b809f9c973a44abb9735facd3f7d0d5da78364a",
    "tokenizer_config.json": "13d6f251e75c2dee803766afb09b03e9a1e5584a45499b042b59de1c82182256",
    "model.safetensors": "15aa3e942aa2d0c610f5ea3dc6ea123980c5e226ff5529950c4dad842299ead2",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str) -> None:
    tmp = DEST / (name + ".part")
    with urllib.request.urlopen(f"{BASE}/{name}") as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total > 50e6:
                print(f"\r  {name}: {done / 1e6:5.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    if total > 50e6:
        print()
    tmp.replace(DEST / name)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = DEST / name
        if path.exists() and sha256(path) == digest:
            print(f"  {name}: already present")
            continue
        print(f"  {name}: downloading")
        download(name)
        if sha256(path) != digest:
            path.unlink()
            print(f"Checksum mismatch for {name}: the download was incomplete. Run the script again.")
            return 1
    print(f"Checker weights ready in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
