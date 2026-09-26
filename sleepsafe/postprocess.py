"""Per-second probabilities -> incident list + session summary (the JSON the agent receives)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.ndimage import median_filter

from .encoder import AUDIOSET_CLASSES

SCORE_NAMES = list(AUDIOSET_CLASSES)


@dataclass
class PostConfig:
    thr_apnea: float = 0.5
    thr_hypopnea: float = 0.5
    thr_snore: float = 0.5
    thr_asleep: float = 0.5
    median_s: int = 5
    merge_gap_s: int = 3
    min_resp_s: int = 10            # clinical minimum for apnea / hypopnea
    min_snore_s: int = 5
    snore_episode_gap_s: int = 30   # snore runs closer than this form one episode
    gate_on_sleep: bool = True      # drop respiratory events during predicted wake
    # zero-shot AudioSet flags: (class, threshold, min seconds)
    zero_shot: dict = field(default_factory=lambda: {
        "cough": ("cough", 0.5, 1), "gasp": ("gasp", 0.4, 1), "snort": ("snort", 0.6, 1),
        "wheeze": ("wheeze", 0.5, 2), "speech": ("speech", 0.7, 5),
    })


def runs(mask: np.ndarray, merge_gap: int = 0, min_len: int = 1) -> list[tuple[int, int]]:
    """Boolean per-second mask -> [(start, end_exclusive)] after gap merging and length filter."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    segs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    merged: list[list[int]] = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged if b - a >= min_len]


def detect_events(probs: np.ndarray, cfg: PostConfig, scores: np.ndarray | None = None) -> list[dict]:
    """probs (T, 4) [apnea, hypopnea, snore, asleep]; scores (T, 12) AudioSet (optional)."""
    p = median_filter(probs, size=(cfg.median_s, 1), mode="nearest") if cfg.median_s > 1 else probs
    asleep = p[:, 3] >= cfg.thr_asleep
    events: list[dict] = []

    def add(kind, a, b, conf, basis="trained", **extra):
        events.append({"type": kind, "start_offset_s": a, "end_offset_s": b, "duration_s": b - a,
                       "confidence": round(float(conf), 3), "confidence_basis": basis, **extra})

    apnea = p[:, 0] >= cfg.thr_apnea
    hyp = (p[:, 1] >= cfg.thr_hypopnea) & ~apnea
    if cfg.gate_on_sleep:
        apnea &= asleep
        hyp &= asleep
    for a, b in runs(apnea, cfg.merge_gap_s, cfg.min_resp_s):
        add("apnea", a, b, p[a:b, 0].mean())
    for a, b in runs(hyp, cfg.merge_gap_s, cfg.min_resp_s):
        if not any(e["type"] == "apnea" and e["start_offset_s"] < b and a < e["end_offset_s"] for e in events):
            add("hypopnea", a, b, p[a:b, 1].mean())
    snore = np.zeros(len(p), bool)
    for a, b in runs(p[:, 2] >= cfg.thr_snore, cfg.merge_gap_s, cfg.min_snore_s):
        snore[a:b] = True
    for a, b in runs(snore, cfg.snore_episode_gap_s, cfg.min_snore_s):
        add("snore_episode", a, b, p[a:b, 2].mean())
    for a, b in runs(~asleep, 0, 60):
        add("wake_period", a, b, 1 - p[a:b, 3].mean())

    if scores is not None:
        for kind, (cls, thr, min_len) in cfg.zero_shot.items():
            col = scores[:, SCORE_NAMES.index(cls)]
            for a, b in runs(col >= thr, 1, min_len):
                add(kind, a, b, col[a:b].max(), basis="zero_shot")
        # link gasps/snorts that terminate an apnea (within 5 s of its end)
        for e in events:
            if e["type"] in ("apnea", "hypopnea"):
                for g in events:
                    if g["type"] in ("gasp", "snort") and 0 <= g["start_offset_s"] - e["end_offset_s"] + 3 <= 8:
                        e.setdefault("evidence", {})["terminated_by"] = g["type"]
    events.sort(key=lambda e: (e["start_offset_s"], e["type"]))
    return events


def signal_quality(energy: np.ndarray, valid_audio: np.ndarray | None = None) -> list[tuple[int, int, str]]:
    """Flag stretches of digital silence / dropouts (≥ 5 s) from band energies (T, 24)."""
    level = energy[:, :8].mean(1)
    dead = level < -90
    if valid_audio is not None:
        dead |= ~valid_audio
    return [(a, b, "no signal / dropout") for a, b in runs(dead, 2, 5)]


def session_summary(session_id: str, start_utc: datetime, probs: np.ndarray, events: list[dict],
                    quality: list[tuple[int, int, str]], cfg: PostConfig, model_info: dict) -> dict:
    T = len(probs)
    bad = np.zeros(T, bool)
    for a, b, _ in quality:
        bad[a:b] = True
    asleep = (median_filter(probs[:, 3], size=cfg.median_s, mode="nearest") >= cfg.thr_asleep) & ~bad
    sleep_s = int(asleep.sum())
    valid_s = int((~bad).sum())
    resp = [e for e in events if e["type"] in ("apnea", "hypopnea")]
    denom_h = (sleep_s if cfg.gate_on_sleep else valid_s) / 3600
    ahi = len(resp) / denom_h if denom_h > 0 else None
    counts: dict[str, int] = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    snore_s = sum(e["duration_s"] for e in events if e["type"] == "snore_episode")
    short = T < 4 * 3600

    for i, e in enumerate(events, 1):
        e["id"] = f"e{i:03d}"
        e["start_utc"] = (start_utc + timedelta(seconds=e["start_offset_s"])).isoformat()
        e["end_utc"] = (start_utc + timedelta(seconds=e["end_offset_s"])).isoformat()
    for a, b, reason in quality:
        events.append({"id": f"q{a}", "type": "low_signal", "start_offset_s": a, "end_offset_s": b,
                       "duration_s": b - a, "reason": reason})

    severity = None
    if ahi is not None:
        severity = "normal" if ahi < 5 else "mild" if ahi < 15 else "moderate" if ahi < 30 else "severe"
    caveats = [
        "Audio-only screening output; not a diagnosis. Hypopnea detection is limited without SpO2.",
        "Model trained on hospital PSG microphones (ambient + tracheal) with lapel-mic augmentation; "
        "not clinically validated on lapel-mic recordings.",
        "zero_shot events (cough, gasp, snort, wheeze, speech) use uncalibrated pretrained-classifier thresholds.",
    ]
    if short:
        caveats.append(f"Short session ({T / 60:.0f} min): the events-per-hour figure is a rough "
                       "extrapolation and should not be read as a clinical AHI.")
    if not cfg.gate_on_sleep:
        caveats.append("Events were counted regardless of sleep/wake state (demo mode); "
                       "rate is per hour of valid recording.")
    return {
        "session_id": session_id,
        "model": model_info,
        "recording": {
            "start_utc": start_utc.isoformat(),
            "end_utc": (start_utc + timedelta(seconds=T)).isoformat(),
            "duration_s": T,
            "valid_audio_s": valid_s,
            "estimated_sleep_s": sleep_s,
        },
        "summary": {
            "respiratory_events_per_hour": None if ahi is None else round(ahi, 1),
            "rate_basis": "per hour of estimated sleep" if cfg.gate_on_sleep else "per hour of valid recording",
            "severity_estimate": severity,
            "severity_bands": "<5 normal, 5-15 mild, 15-30 moderate, >=30 severe",
            "short_session": short,
            "counts": counts,
            "snore_pct_of_sleep": round(100 * snore_s / sleep_s, 1) if sleep_s else None,
            "longest_apnea_s": max((e["duration_s"] for e in events if e["type"] == "apnea"), default=0),
        },
        "events": events,
        "caveats": caveats,
    }


def config_to_dict(cfg: PostConfig) -> dict:
    return asdict(cfg)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
