"""Stage 2: preprocessing + BiGRU head over per-second Stage-1 features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import TARGETS

WINDOW_S = 600   # training crop / inference window (10 min)
MARGIN_S = 60    # inference: discard this much at each window edge (except session edges)
NORM_BLOCK_S = 1800  # energy features are referenced to the median of each 30-min block


def block_median(e: torch.Tensor, block: int = NORM_BLOCK_S) -> torch.Tensor:
    """Per-block median over time (T, C) -> (T, C). A trailing block shorter than half a
    block is merged into the previous one, so a 30-min session is always one block."""
    T = e.shape[0]
    bounds = list(range(0, T, block)) + [T]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < block // 2:
        bounds.pop(-2)
    out = torch.empty_like(e)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = e[a:b].median(dim=0, keepdim=True).values
    return out


class Preprocessor(nn.Module):
    """Raw Stage-1 features of one session -> model input (T, D).

    emb (T, 2048) -> PCA to `pca_dim`; scores (T, 12) -> logit; energy (T, 24) -> minus the
    median of its 30-min block (removes mic gain / distance; a 30-min session is one block).
    Everything is then standardized with training-set statistics. All parameters are fixed
    buffers fitted before training.
    """

    def __init__(self, pca_dim: int = 256, n_scores: int = 12, n_energy: int = 24):
        super().__init__()
        self.register_buffer("pca_mean", torch.zeros(2048))
        self.register_buffer("pca_w", torch.zeros(2048, pca_dim))
        d = pca_dim + n_scores + n_energy
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))
        self.dim = d

    def raw(self, emb, scores, energy) -> torch.Tensor:
        z = (emb.float() - self.pca_mean) @ self.pca_w
        s = torch.logit(scores.float().clamp(1e-4, 1 - 1e-4))
        e = energy.float()
        # mean/max band levels become relative to the block; std columns are already level-free
        e = torch.cat([e[:, :8] - block_median(e[:, :8]), e[:, 8:16] - block_median(e[:, 8:16]), e[:, 16:]], -1)
        return torch.cat([z, s, e], dim=-1)

    def forward(self, emb, scores, energy) -> torch.Tensor:
        return (self.raw(emb, scores, energy) - self.mu) / self.sd

    @torch.no_grad()
    def fit_pca(self, emb_sample: torch.Tensor) -> float:
        x = emb_sample.float()
        self.pca_mean.copy_(x.mean(0))
        cov = torch.cov((x - self.pca_mean).T)
        evals, evecs = torch.linalg.eigh(cov)
        order = evals.argsort(descending=True)[: self.pca_w.shape[1]]
        self.pca_w.copy_(evecs[:, order])
        return (evals[order].sum() / evals.sum()).item()

    @torch.no_grad()
    def fit_norm(self, raw_sample: torch.Tensor) -> None:
        self.mu.copy_(raw_sample.mean(0))
        self.sd.copy_(raw_sample.std(0).clamp_min(1e-3))


class SleepHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.conv = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, len(TARGETS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, D) -> logits (B, T, 4)."""
        h = self.inp(x)
        h = h + F.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        h, _ = self.gru(h)
        return self.out(h)


@torch.no_grad()
def predict_session(head: SleepHead, x: torch.Tensor, batch: int = 32) -> torch.Tensor:
    """x (T, D) for a whole session -> per-second probabilities (T, 4).

    Overlapping WINDOW_S windows; each second is taken from the window where it is
    furthest from an edge. Sessions shorter than a window run in one pass.
    """
    head.eval()
    x = x.float()
    T = x.shape[0]
    if T <= WINDOW_S:
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            return torch.sigmoid(head(x.unsqueeze(0)).float())[0]
    stride = WINDOW_S - 2 * MARGIN_S
    starts = list(range(0, T - WINDOW_S, stride)) + [T - WINDOW_S]
    out = torch.zeros(T, len(TARGETS), device=x.device)
    wins = torch.stack([x[s: s + WINDOW_S] for s in starts])
    probs = []
    for i in range(0, len(starts), batch):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            probs.append(torch.sigmoid(head(wins[i: i + batch]).float()))
    probs = torch.cat(probs)
    for i, s in enumerate(starts):
        a = 0 if i == 0 else MARGIN_S
        b = WINDOW_S if i == len(starts) - 1 else WINDOW_S - MARGIN_S
        # last window overlaps the previous one irregularly: only fill what it owns
        lo = s + a if i < len(starts) - 1 else max(s + a, starts[i - 1] + WINDOW_S - MARGIN_S if i else 0)
        out[lo: s + b] = probs[i, lo - s: b]
    return out
