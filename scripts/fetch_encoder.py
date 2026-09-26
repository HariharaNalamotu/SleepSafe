"""Download the frozen PANNs CNN14 (16 kHz) encoder weights to data/pretrained/Cnn14_16k.pth.

The file (358 MB) is too large for GitHub, so it is fetched from a Hugging Face mirror.

    python scripts/fetch_encoder.py
"""
from pathlib import Path

import requests

URL = "https://huggingface.co/niobures/PANNs/resolve/main/models/pretrained/Cnn14_16k_mAP%3D0.438.pth"
SIZE = 358668570
DEST = Path(__file__).resolve().parents[1] / "data" / "pretrained" / "Cnn14_16k.pth"


def main():
    if DEST.exists() and DEST.stat().st_size == SIZE:
        print(f"already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    if tmp.stat().st_size != SIZE:
        raise RuntimeError(f"unexpected size {tmp.stat().st_size} (expected {SIZE})")
    tmp.replace(DEST)
    print(f"saved {DEST}")


if __name__ == "__main__":
    main()
