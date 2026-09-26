"""Download a subset of PSG-Audio (V3) from the Hugging Face mirror and convert it.

For every 1-hour EDF part we keep only the two audio channels (Mic, Tracheal),
resampled 48 kHz -> 16 kHz and stored as int16 .npy, then delete the EDF.
All V3 RML annotation files are downloaded as well.

Output layout:
    data/rml/<subject>.rml
    data/audio16k/<subject>/<part:03d>_mic.npy
    data/audio16k/<subject>/<part:03d>_trach.npy
    data/subjects.json          (selected subjects, in order)

Resumable: parts whose .npy outputs already exist are skipped.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from scipy.signal import resample_poly

REPO = "https://huggingface.co/datasets/dust-systems/psg-audio/resolve/main/"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AUDIO_CHANNELS = {"Mic": "mic", "Tracheal": "trach"}


def url_for(path: str) -> str:
    # File names contain a literal "%5B001%5D", so the "%" itself must be escaped.
    return REPO + urllib.parse.quote(path)


def fetch(url: str, dest: Path, retries: int = 5) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - retry any network failure
            print(f"  retry {attempt + 1}/{retries} {dest.name}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def read_edf_audio(path: Path) -> dict[str, np.ndarray]:
    """Return {'mic': int16[ n ], 'trach': int16[ n ]} at the native 48 kHz."""
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256])
        hdr_bytes = int(head[184:192])
        n_rec = int(head[236:244])
        sig = f.read(ns * 256)
    labels = [sig[i * 16:(i + 1) * 16].decode().strip() for i in range(ns)]
    off = ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    spr = [int(sig[off + i * 8: off + (i + 1) * 8]) for i in range(ns)]
    rec_len = sum(spr)
    raw = np.memmap(path, dtype="<i2", mode="r", offset=hdr_bytes)
    n_rec = min(n_rec, raw.size // rec_len) if n_rec > 0 else raw.size // rec_len
    raw = raw[: n_rec * rec_len].reshape(n_rec, rec_len)
    starts = np.cumsum([0] + spr[:-1])
    out = {}
    for name, key in AUDIO_CHANNELS.items():
        i = labels.index(name)
        out[key] = np.ascontiguousarray(raw[:, starts[i]: starts[i] + spr[i]]).reshape(-1)
    return out


def to_16k(x: np.ndarray) -> np.ndarray:
    y = resample_poly(x.astype(np.float32), 1, 3)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def process_part(subject: str, part: int, path_in_repo: str) -> str:
    out_dir = DATA / "audio16k" / subject
    outs = {k: out_dir / f"{part:03d}_{k}.npy" for k in AUDIO_CHANNELS.values()}
    if all(p.exists() for p in outs.values()):
        return f"skip {subject}[{part}]"
    out_dir.mkdir(parents=True, exist_ok=True)
    edf = DATA / "raw" / f"{subject}_{part:03d}.edf"
    t0 = time.time()
    if not edf.exists():
        fetch(url_for(path_in_repo), edf)
    t1 = time.time()
    audio = read_edf_audio(edf)
    for key, x in audio.items():
        tmp = outs[key].with_suffix(".tmp.npy")
        np.save(tmp, to_16k(x))
        tmp.replace(outs[key])
    edf.unlink()
    return f"done {subject}[{part}] dl {t1 - t0:.0f}s conv {time.time() - t1:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    files = [
        f.path
        for f in HfApi().list_repo_tree(
            "dust-systems/psg-audio", repo_type="dataset", path_in_repo="V3", recursive=True
        )
        if f.__class__.__name__ == "RepoFile"
    ]
    rml = {p.split("/")[-1][:-4]: p for p in files if "/APNEA_RML/" in p and p.endswith(".rml")}

    (DATA / "rml").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda kv: (DATA / "rml" / f"{kv[0]}.rml").exists()
                    or fetch(url_for(kv[1]), DATA / "rml" / f"{kv[0]}.rml"), rml.items()))
    print(f"{len(rml)} RML files ready", flush=True)

    parts: dict[str, dict[int, str]] = {}
    for p in files:
        if p.endswith(".edf"):
            s = p.split("/")[2]
            n = int(re.search(r"%5B(\d+)%5D", p).group(1))
            parts.setdefault(s, {})[n] = p
    complete = sorted(
        s for s, v in parts.items() if s in rml and sorted(v) == list(range(1, len(v) + 1))
    )

    sel_file = DATA / "subjects.json"
    if sel_file.exists():
        selected = json.loads(sel_file.read_text())
    else:
        selected = random.Random(args.seed).sample(complete, min(args.n_subjects, len(complete)))
        sel_file.write_text(json.dumps(selected, indent=1))
    print(f"{len(complete)} complete subjects, downloading {len(selected)}", flush=True)

    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    jobs = [(s, n, parts[s][n]) for s in selected for n in sorted(parts[s])]
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_part, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(jobs)}] {fut.result()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(jobs)}] ERROR {e}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
