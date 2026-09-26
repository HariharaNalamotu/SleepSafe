"""GPU waveform augmentations that mimic a lapel mic worn during sleep.

Applied per 64 s segment (B, S) before the frozen encoder:
    gain        ±9 dB
    room noise  pink/brown noise at 15–40 dB below the segment level
    muffling    low-pass 1.5–4 kHz (mic covered by bedding)
    rustle      bursts of high-passed noise with jittery envelopes (clothing friction)
    dropout     short digital silences (mic/radio glitches)
"""
from __future__ import annotations

import torch

from .encoder import SR


def _colored_noise(shape, exponent: torch.Tensor, g: torch.Generator, device) -> torch.Tensor:
    """1/f^exponent noise, unit RMS. exponent: (B,) — 1 = pink, 2 = brown."""
    white = torch.randn(shape, generator=g, device=device)
    spec = torch.fft.rfft(white)
    f = torch.fft.rfftfreq(shape[-1], 1 / SR, device=device).clamp_min(20.0)
    spec = spec / f.unsqueeze(0).pow(exponent.unsqueeze(1) / 2)
    x = torch.fft.irfft(spec, n=shape[-1])
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def _lowpass(x: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(x.shape[-1], 1 / SR, device=x.device)
    # 4th-order-like smooth rolloff
    h = 1.0 / (1.0 + (f.unsqueeze(0) / cutoff.unsqueeze(1)).pow(8))
    return torch.fft.irfft(spec * h.sqrt(), n=x.shape[-1])


class LapelAugment:
    def __init__(self, seed: int, device="cuda"):
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    def _u(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=self.g, device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        level = x.pow(2).mean(-1).sqrt().clamp_min(1e-5)           # (B,)

        # Muffling
        muff = torch.rand(B, generator=self.g, device=self.device) < 0.3
        if muff.any():
            x[muff] = _lowpass(x[muff], self._u(int(muff.sum()), 1500, 4000))

        # Stationary room / fabric noise
        noise = _colored_noise((B, S), self._u(B, 0.8, 2.0), self.g, self.device)
        snr_db = self._u(B, 15, 40)
        x = x + noise * (level * 10 ** (-snr_db / 20)).unsqueeze(1)

        # Rustle bursts
        n_burst = torch.randint(0, 4, (B,), generator=self.g, device=self.device)
        hp = torch.fft.rfftfreq(S, 1 / SR, device=self.device) > 800
        for b in range(B):
            for _ in range(int(n_burst[b])):
                dur = int(self._u(1, 0.3, 4.0).item() * SR)
                start = int(torch.randint(0, S - dur, (1,), generator=self.g, device=self.device))
                burst = torch.randn(dur, generator=self.g, device=self.device)
                burst = torch.fft.irfft(torch.fft.rfft(burst) * hp[: dur // 2 + 1], n=dur)
                env = torch.hann_window(dur, device=self.device)
                jitter = torch.rand(dur // 400 + 1, generator=self.g, device=self.device)
                env = env * torch.repeat_interleave(jitter, 400)[:dur]
                amp = level[b] * 10 ** (self._u(1, -5, 15) / 20)
                x[b, start:start + dur] += amp * env * burst / burst.std().clamp_min(1e-8)

        # Dropouts
        drop = torch.rand(B, generator=self.g, device=self.device) < 0.1
        for b in torch.nonzero(drop).flatten().tolist():
            dur = int(self._u(1, 0.3, 2.0).item() * SR)
            start = int(torch.randint(0, S - dur, (1,), generator=self.g, device=self.device))
            x[b, start:start + dur] = 0

        gain = 10 ** (self._u(B, -9, 9) / 20)
        return (x * gain.unsqueeze(1)).clamp(-1, 1)
