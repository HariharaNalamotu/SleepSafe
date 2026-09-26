"""Stage-1 feature extraction for every downloaded PSG-Audio subject (GPU).

Per subject, 6 variants are encoded:
    v0 mic   v1 tracheal                (clean)
    v2 mic   v3 tracheal                (lapel augmentation, seed A)
    v4 mic   v5 tracheal                (lapel augmentation, seed B)

Output: data/features/<subject>/{emb,scores,energy}_v<k>.npy, labels.npz
Runs alongside download_convert.py and waits for subjects to finish downloading.
"""
import json
import re
import sys
import threading
import time
import zlib
from pathlib import Path
from queue import Queue

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleepsafe.augment import LapelAugment  # noqa: E402
from sleepsafe.encoder import Cnn14Encoder, encode, normalize_loudness  # noqa: E402
from sleepsafe.labels import night_labels  # noqa: E402

DATA = ROOT / "data"
VARIANTS = [("mic", None), ("trach", None), ("mic", 1), ("trach", 2), ("mic", 3), ("trach", 4)]


def expected_parts() -> dict[str, int]:
    files = json.loads((ROOT / "scratch" / "v3_files.json").read_text())
    counts: dict[str, int] = {}
    for f in files:
        if f["path"].endswith(".edf"):
            s = f["path"].split("/")[2]
            counts[s] = max(counts.get(s, 0), int(re.search(r"%5B(\d+)%5D", f["path"]).group(1)))
    return counts


def load_subject(s: str, n_parts: int) -> dict[str, np.ndarray]:
    d = DATA / "audio16k" / s
    return {ch: np.concatenate([np.load(d / f"{p:03d}_{ch}.npy") for p in range(1, n_parts + 1)])
            for ch in ("mic", "trach")}


def main() -> None:
    # Low-VRAM mode: sustained runs that filled the 16 GB (and spilled into shared system
    # memory) coincided with system crashes. Cap PyTorch well below the card's capacity so it
    # errors instead of spilling, use small batches and skip cuDNN autotuning workspaces.
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.6)
    subjects = json.loads((DATA / "subjects.json").read_text())
    n_parts = expected_parts()
    model = Cnn14Encoder(str(DATA / "pretrained" / "Cnn14_16k.pth")).cuda()

    def ready(s):
        d = DATA / "audio16k" / s
        return all((d / f"{p:03d}_{ch}.npy").exists() for p in range(1, n_parts[s] + 1) for ch in ("mic", "trach"))

    todo = [s for s in subjects if not (DATA / "features" / s / "labels.npz").exists()]
    q: Queue = Queue(maxsize=2)

    def producer():
        pending = list(todo)
        while pending:
            s = next((s for s in pending if ready(s)), None)
            if s is None:
                time.sleep(10)
                continue
            pending.remove(s)
            q.put((s, load_subject(s, n_parts[s])))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        s, audio = item
        t0 = time.time()
        out = DATA / "features" / s
        out.mkdir(parents=True, exist_ok=True)
        lab = night_labels(DATA / "rml" / f"{s}.rml")
        wavs = {ch: normalize_loudness(torch.from_numpy(a).cuda().float() / 32768) for ch, a in audio.items()}
        T = min(lab.duration_s, min(w.shape[0] for w in wavs.values()) // 16000)
        for k, (ch, seed) in enumerate(VARIANTS):
            aug = LapelAugment(seed=zlib.crc32(f"{s}/{seed}".encode())) if seed is not None else None
            f = encode(model, wavs[ch], batch=8, augment=aug)
            np.save(out / f"emb_v{k}.npy", f["emb"][:T].half().cpu().numpy())
            np.save(out / f"scores_v{k}.npy", f["scores"][:T].half().cpu().numpy())
            np.save(out / f"energy_v{k}.npy", f["energy"][:T].cpu().numpy())
            del f
            torch.cuda.empty_cache()
        events = np.array([(e.start, e.duration, {"apnea": 0, "hypopnea": 1, "snore": 2}[
            "snore" if e.type == "Snore" else "hypopnea" if e.type == "Hypopnea" else "apnea"])
            for e in lab.events], np.float32).reshape(-1, 3)
        np.savez(out / "labels.npz", y=lab.y[:T], mask=lab.mask[:T], events=events, ahi=lab.ahi)
        print(f"{s}: T={T}s ({T / 3600:.1f}h) AHI={lab.ahi:.1f} in {time.time() - t0:.0f}s, "
              f"peak GPU {torch.cuda.max_memory_reserved() / 1e9:.1f} GB", flush=True)
        del wavs
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
