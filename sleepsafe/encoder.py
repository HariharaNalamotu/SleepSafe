"""Stage 1: frozen PANNs CNN14 (16 kHz) → per-second features, fully on GPU.

For every second of audio this produces:
    emb     2048-d  CNN14 frame embeddings (fc1 + ReLU), averaged within the second
    scores  12      AudioSet probabilities for sleep-relevant classes, max within the second
    energy  24      log energy in 8 mel bands: mean / max / std within the second

Audio is processed in 64 s segments with 3.2 s of context on each side so that
segment boundaries do not affect the output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
HOP = 160                      # 10 ms
N_FFT = 512
FRAMES_PER_S = SR // HOP       # 100 mel frames per second
CORE_S = 64                    # seconds of output per segment
CTX_S = 3.2                    # context on each side (= 320 mel frames = 10 CNN frames)
CTX = int(CTX_S * SR)
SEG = CORE_S * SR + 2 * CTX

AUDIOSET_CLASSES = {
    "speech": 0, "breathing": 41, "wheeze": 42, "snoring": 43, "gasp": 44, "pant": 45,
    "snort": 46, "cough": 47, "throat_clearing": 48, "sniff": 50, "rustle": 487, "silence": 500,
}
N_BANDS = 8


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x, pool):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool) if pool != (1, 1) else x


class Cnn14Encoder(nn.Module):
    def __init__(self, checkpoint: str | None = None):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("melW", torch.zeros(N_FFT // 2 + 1, 64))
        self.bn0 = nn.BatchNorm2d(64)
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", ConvBlock(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)
        self.register_buffer("class_idx", torch.tensor(list(AUDIOSET_CLASSES.values())), persistent=False)
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            sd = {k.replace("logmel_extractor.", ""): v for k, v in sd.items()
                  if not k.startswith("spectrogram_extractor.")}
            self.load_state_dict(sd)
        self.eval()

    def logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, S) float in [-1, 1] -> log-mel dB (B, frames, 64), float32."""
        spec = torch.stft(wav.float(), N_FFT, HOP, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True)
        power = spec.real.square() + spec.imag.square()           # (B, 257, frames)
        mel = torch.matmul(power.transpose(1, 2), self.melW)       # (B, frames, 64)
        return 10.0 * torch.log10(mel.clamp_min(1e-10))

    @torch.inference_mode()
    def forward(self, seg: torch.Tensor) -> dict[str, torch.Tensor]:
        """seg (B, SEG) float32 -> per-second features for the CORE_S seconds of each segment."""
        B = seg.shape[0]
        lm = self.logmel(seg)                                      # (B, 7041, 64)

        # Hand-crafted band energies from the core region.
        c0 = int(CTX_S * FRAMES_PER_S)
        core = lm[:, c0: c0 + CORE_S * FRAMES_PER_S]               # (B, 6400, 64)
        band = torch.logsumexp(core.view(B, -1, N_BANDS, 64 // N_BANDS) * (0.1 * 2.302585), dim=-1)
        band = band / (0.1 * 2.302585) - 10 * torch.log10(torch.tensor(64 / N_BANDS, device=seg.device))
        band = band.view(B, CORE_S, FRAMES_PER_S, N_BANDS)
        energy = torch.cat([band.mean(2), band.amax(2), band.std(2)], dim=-1)   # (B, 64, 24)

        with torch.autocast(seg.device.type, dtype=torch.bfloat16, enabled=seg.is_cuda):
            x = lm.unsqueeze(1)                                    # (B, 1, frames, 64)
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
            x = x.contiguous(memory_format=torch.channels_last)
            for i in range(1, 6):
                x = getattr(self, f"conv_block{i}")(x, (2, 2))
            x = self.conv_block6(x, (1, 1))                        # (B, 2048, T', 2)
            x = x.mean(dim=3)                                      # (B, 2048, T')
            x = F.max_pool1d(x, 3, 1, 1) + F.avg_pool1d(x, 3, 1, 1)
            ctx = round(CTX_S * FRAMES_PER_S / 32)                 # 10 frames
            x = x[:, :, ctx: x.shape[-1] - ctx]                    # (B, 2048, 200)
            # fc1 was trained on (max + mean) over time ≈ 2x a single frame.
            h = F.relu(self.fc1(2.0 * x.transpose(1, 2)))           # (B, 200, 2048)
            logits = self.fc_audioset(h)[..., self.class_idx]      # (B, 200, 12)
        h = h.float().transpose(1, 2)
        emb = F.adaptive_avg_pool1d(h, CORE_S).transpose(1, 2)     # (B, 64, 2048)
        probs = torch.sigmoid(logits.float()).transpose(1, 2)
        scores = F.adaptive_max_pool1d(probs, CORE_S).transpose(1, 2)  # (B, 64, 12)
        return {"emb": emb, "scores": scores, "energy": energy}


TARGET_P90_DBFS = -20.0
MAX_GAIN_DB = 40.0


def normalize_loudness(wav: torch.Tensor) -> torch.Tensor:
    """Scale a whole session so its 90th-percentile per-second RMS sits at -20 dBFS.

    Makes features independent of mic gain / placement; applied identically in
    training and in deployment (per session, after the session ends).
    """
    n = wav.shape[0] // SR * SR
    rms = wav[:n].view(-1, SR).pow(2).mean(1).sqrt()
    ref = torch.quantile(rms[:: max(1, rms.numel() // 100_000)], 0.9).clamp_min(1e-6)
    gain = (10 ** (TARGET_P90_DBFS / 20) / ref).clamp(max=10 ** (MAX_GAIN_DB / 20))
    return (wav * gain).clamp(-1, 1)


def segment(wav: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Split a 1-D waveform into overlapping (n, SEG) segments; returns (segments, n_seconds)."""
    n_s = wav.shape[0] // SR
    n_seg = -(-n_s // CORE_S)
    total = n_seg * CORE_S * SR
    padded = F.pad(wav[: n_s * SR], (CTX, total - n_s * SR + CTX))
    segs = padded.unfold(0, SEG, CORE_S * SR)                      # (n_seg, SEG) view
    return segs, n_s


@torch.inference_mode()
def encode(model: Cnn14Encoder, wav: torch.Tensor, batch: int = 24, augment=None) -> dict[str, torch.Tensor]:
    """wav: 1-D float tensor on GPU in [-1, 1]. Returns per-second features (T, ·) on GPU."""
    segs, n_s = segment(wav)
    outs = {"emb": [], "scores": [], "energy": []}
    for i in range(0, segs.shape[0], batch):
        x = segs[i: i + batch]
        if augment is not None:
            x = augment(x.clone())
        o = model(x)
        for k in outs:
            outs[k].append(o[k].reshape(-1, o[k].shape[-1]))
    return {k: torch.cat(v)[:n_s] for k, v in outs.items()}
