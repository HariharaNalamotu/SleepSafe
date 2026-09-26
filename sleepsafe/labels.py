"""Per-second training targets from PSG-Audio RML annotations.

Targets (float32, one value per second of recording):
    apnea     obstructive / mixed / central apnea (technician scored)
    hypopnea  hypopnea (technician scored)
    snore     snore events (device scored)
    asleep    1 for N1/N2/N3/REM, 0 for Wake

Masks (bool, True = include in the loss):
    resp_mask   asleep & not within EDGE_S of a respiratory event boundary
                (events are only scored during sleep; onsets come from the
                airflow sensor so the acoustic boundary is fuzzy)
    snore_mask  not within EDGE_S of a snore event boundary
    stage_mask  epoch was scored
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NS = {"p": "http://www.respironics.com/PatientStudy.xsd"}
APNEA_TYPES = {"ObstructiveApnea", "MixedApnea", "CentralApnea"}
SLEEP_STAGES = {"NonREM1", "NonREM2", "NonREM3", "REM"}
EDGE_S = 2
TARGETS = ("apnea", "hypopnea", "snore", "asleep")


@dataclass
class Event:
    family: str
    type: str
    start: float
    duration: float


@dataclass
class NightLabels:
    duration_s: int
    y: np.ndarray        # (T, 4) float32 in TARGETS order
    mask: np.ndarray     # (T, 4) bool
    events: list[Event]  # respiratory + snore events, for event-level evaluation

    @property
    def ahi(self) -> float:
        """Apneas + hypopneas per hour of scored sleep (clinical definition)."""
        n = sum(e.type in APNEA_TYPES or e.type == "Hypopnea" for e in self.events)
        tst_h = self.y[:, 3][self.mask[:, 3]].sum() / 3600
        return n / tst_h if tst_h > 0 else float("nan")


def parse_rml(path: str | Path) -> tuple[int, list[Event], list[tuple[float, str]]]:
    root = ET.parse(path).getroot()
    duration = int(float(root.find("p:Acquisition/p:Sessions/p:Session/p:Duration", NS).text))
    events = [
        Event(e.get("Family"), e.get("Type"), float(e.get("Start")), float(e.get("Duration")))
        for e in root.iter(f"{{{NS['p']}}}Event")
    ]
    staging = root.find("p:ScoringData/p:StagingData/p:UserStaging/p:NeuroAdultAASMStaging", NS)
    if staging is None:
        staging = root.find(".//p:NeuroAdultAASMStaging", NS)
    stages = [(float(s.get("Start")), s.get("Type")) for s in staging.iter(f"{{{NS['p']}}}Stage")]
    return duration, events, sorted(stages)


def _paint(arr: np.ndarray, start: float, dur: float, value=1.0) -> None:
    """Mark seconds whose majority lies inside [start, start + dur)."""
    a = int(np.floor(start + 0.5))
    b = int(np.floor(start + dur + 0.5))
    arr[max(a, 0): max(b, 0)] = value


def _edges(mask: np.ndarray, start: float, dur: float) -> None:
    for t in (start, start + dur):
        c = int(round(t))
        mask[max(c - EDGE_S, 0): max(c + EDGE_S, 0)] = False


def night_labels(path: str | Path) -> NightLabels:
    duration, events, stages = parse_rml(path)
    T = duration
    y = np.zeros((T, len(TARGETS)), np.float32)
    mask = np.ones((T, len(TARGETS)), bool)

    # Sleep stages: each stage holds until the next change point.
    bounds = [s for s, _ in stages] + [T]
    for (start, stage), end in zip(stages, bounds[1:]):
        a, b = int(start), int(end)
        if stage in SLEEP_STAGES:
            y[a:b, 3] = 1
        elif stage != "Wake":  # NotScored / QuietSleep / unknown
            mask[a:b, 3] = False
    if stages and stages[0][0] > 0:
        mask[: int(stages[0][0]), 3] = False

    kept = []
    for e in events:
        if e.type in APNEA_TYPES:
            col = 0
        elif e.type == "Hypopnea":
            col = 1
        elif e.family == "Nasal" and e.type == "Snore":
            col = 2
        else:
            continue
        kept.append(e)
        _paint(y[:, col], e.start, e.duration)
        cols = (0, 1) if col < 2 else (2,)
        for c in cols:
            _edges(mask[:, c], e.start, e.duration)

    asleep_known = mask[:, 3] & (y[:, 3] > 0)
    mask[:, 0] &= asleep_known
    mask[:, 1] &= asleep_known
    return NightLabels(T, y, mask, kept)
