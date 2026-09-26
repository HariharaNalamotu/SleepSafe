"""WAV inspection, waveform standardization, quality checks, and windowing.

The low-signal threshold is an audio-quality heuristic only. It is not a
clinical threshold and must not be interpreted as a health finding.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


# A small digital silence guard catches files that contain only quantization
# noise. The separate low-signal threshold is intentionally conservative and
# is only used to flag recording quality.
_SILENCE_PEAK_THRESHOLD = 1e-8
LOW_SIGNAL_RMS_THRESHOLD = 1e-3


def inspect_original_audio(path):
    """Return source WAV metadata without loading the entire recording."""
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError, TypeError) as exc:
        raise ValueError(f"Unable to read WAV file: {path}") from exc
    if info.frames <= 0:
        raise ValueError("Audio file is empty")
    if info.samplerate <= 0 or info.channels <= 0:
        raise ValueError("Audio file has invalid sample rate or channel count")
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_s": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
    }


def standardize_audio(path, target_sr=16000):
    """Load WAV, average channels to mono, and resample to ``target_sr``."""
    if not isinstance(target_sr, int) or isinstance(target_sr, bool) or target_sr <= 0:
        raise ValueError("target_sr must be a positive integer")
    inspect_original_audio(path)
    try:
        audio, original_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, TypeError) as exc:
        raise ValueError(f"Unable to decode WAV file: {path}") from exc
    if audio.size == 0:
        raise ValueError("Audio file is empty")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if original_sr != target_sr:
        mono = librosa.resample(mono, orig_sr=original_sr, target_sr=target_sr)
    mono = np.asarray(mono, dtype=np.float32)
    validate_audio(mono, target_sr)
    return mono


def validate_audio(audio, sr):
    """Raise ``ValueError`` for unusable, non-finite, or silent waveforms."""
    return _validate_audio(audio, sr, require_signal=True)


def _validate_audio(audio, sr, require_signal):
    if not isinstance(sr, (int, np.integer)) or isinstance(sr, (bool, np.bool_)) or sr <= 0:
        raise ValueError("Sample rate must be a positive integer")
    values = np.asarray(audio)
    if values.size == 0 or values.ndim not in (1, 2):
        raise ValueError("Audio is empty or has an unsupported shape")
    if values.ndim == 2 and 1 not in values.shape:
        raise ValueError("Audio must be mono (one-dimensional or one channel)")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Audio samples must be numeric")
    if not np.isfinite(values).all():
        raise ValueError("Audio contains NaN or infinite samples")
    if values.size / int(sr) <= 0:
        raise ValueError("Audio duration must be greater than zero")
    if require_signal and float(np.max(np.abs(values))) <= _SILENCE_PEAK_THRESHOLD:
        raise ValueError("Audio is completely silent")
    return True


def compute_quality_metrics(audio, sr):
    """Return deterministic basic quality metrics for one waveform."""
    # Individual windows can be silent even when the complete recording has
    # usable signal. Keep them and flag quality instead of dropping data.
    _validate_audio(audio, sr, require_signal=False)
    values = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    return {
        "duration_s": float(values.size / int(sr)),
        "peak_amplitude": peak,
        "rms": rms,
        "low_signal": bool(rms < LOW_SIGNAL_RMS_THRESHOLD),
    }


def split_into_windows(audio, sr, window_seconds=1200):
    """Split waveform into sequential windows, retaining a final partial one."""
    validate_audio(audio, sr)
    if not isinstance(window_seconds, (int, float)) or isinstance(window_seconds, bool) or window_seconds <= 0:
        raise ValueError("window_seconds must be a positive number")
    window_frames = int(round(float(window_seconds) * int(sr)))
    if window_frames <= 0:
        raise ValueError("window_seconds is too small for this sample rate")
    values = np.asarray(audio)
    return [values[start:min(start + window_frames, values.size)]
            for start in range(0, values.size, window_frames)]


def save_window(audio, sr, output_path):
    """Save a mono window as 16-bit PCM WAV and return its path."""
    _validate_audio(audio, sr, require_signal=False)
    values = np.asarray(audio)
    if values.ndim == 2:
        values = values.reshape(-1)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), values, int(sr), format="WAV", subtype="PCM_16")
    return str(destination)
