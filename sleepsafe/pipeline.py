"""End-to-end session inference: audio chunks -> night JSON for the report agent.

Used identically on a laptop (GPU) and in the Databricks end-of-session job (CPU).
"""
from __future__ import annotations

import re
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .encoder import SR, Cnn14Encoder, encode, normalize_loudness
from .model import Preprocessor, SleepHead, predict_session
from .postprocess import PostConfig, detect_events, session_summary, signal_quality

MODEL_NAME = "sleepsafe-crnn"


def load_audio(path: str | Path) -> np.ndarray:
    """Any soundfile-readable file -> mono float32 at 16 kHz."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def assemble_chunks(paths: list[str | Path], chunk_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ordered chunk files (named ..._<index>.<ext>) -> (audio, per-second valid mask).

    Missing chunk indices are filled with silence and marked invalid.
    """
    def idx(p):
        m = re.findall(r"(\d+)", Path(p).stem)
        return int(m[-1]) if m else 0

    paths = sorted(paths, key=idx)
    chunks = {idx(p): load_audio(p) for p in paths}
    if chunk_s is None:
        chunk_s = np.median([len(c) for c in chunks.values()]) / SR
    n = int(round(chunk_s * SR))
    first, last = min(chunks), max(chunks)
    parts, valid = [], []
    for i in range(first, last + 1):
        c = chunks.get(i)
        if c is None:
            parts.append(np.zeros(n, np.float32))
            valid.append(np.zeros(n, bool))
        else:
            parts.append(c)
            valid.append(np.ones(len(c), bool))
    audio = np.concatenate(parts)
    v = np.concatenate(valid)
    T = len(audio) // SR
    return audio, v[: T * SR].reshape(T, SR).all(1)


class SleepSafePipeline:
    def __init__(self, encoder_ckpt: str | Path, model_ckpt: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Cnn14Encoder(str(encoder_ckpt)).to(self.device)
        ck = torch.load(model_ckpt, map_location="cpu", weights_only=False)
        self.pre = Preprocessor(pca_dim=ck["pca_dim"])
        self.pre.load_state_dict(ck["preprocessor"])
        self.pre.to(self.device)
        self.head = SleepHead(self.pre.dim, **ck["head_args"])
        self.head.load_state_dict(ck["head"])
        self.head.to(self.device).eval()
        self.post = PostConfig(**{k: v for k, v in ck["post_config"].items()})
        self.version = ck.get("version", "0.1.0")

    @torch.inference_mode()
    def features(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
        wav = normalize_loudness(torch.from_numpy(audio).to(self.device).float())
        return encode(self.encoder, wav, batch=24 if self.device == "cuda" else 4)

    @torch.inference_mode()
    def run(self, audio: np.ndarray, session_id: str, start_utc: datetime,
            valid: np.ndarray | None = None, demo_mode: bool = False) -> dict:
        if len(audio) < 60 * SR:
            raise ValueError("need at least 60 s of audio")
        f = self.features(audio)
        x = self.pre(f["emb"], f["scores"], f["energy"])
        probs = predict_session(self.head, x).float().cpu().numpy()
        scores = f["scores"].float().cpu().numpy()
        energy = f["energy"].float().cpu().numpy()
        cfg = PostConfig(**{**self.post.__dict__, "gate_on_sleep": not demo_mode})
        events = detect_events(probs, cfg, scores)
        quality = signal_quality(energy, None if valid is None else valid[: len(probs)])
        return session_summary(
            session_id, start_utc, probs, events, quality, cfg,
            {"name": MODEL_NAME, "version": self.version, "encoder": "panns-cnn14-16k (frozen)",
             "demo_mode": demo_mode},
        )
