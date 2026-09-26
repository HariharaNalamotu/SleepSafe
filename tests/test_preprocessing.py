import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from preprocessing.audio import (
    compute_quality_metrics,
    inspect_original_audio,
    save_window,
    split_into_windows,
    standardize_audio,
    validate_audio,
)
from preprocessing.manifest import MANIFEST_FIELDS, build_window_manifest


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def wav(self, name, samples, sr):
        path = self.root / name
        sf.write(path, samples, sr, subtype="PCM_16")
        return path

    def test_normal_mono_16khz(self):
        path = self.wav("mono.wav", np.full(16000, 0.1, dtype=np.float32), 16000)
        self.assertEqual(inspect_original_audio(path)["channels"], 1)
        audio = standardize_audio(path)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.size, 16000)

    def test_stereo_converts_to_mono(self):
        samples = np.column_stack((np.full(8000, 0.2), np.full(8000, -0.1))).astype(np.float32)
        path = self.wav("stereo.wav", samples, 8000)
        self.assertEqual(inspect_original_audio(path)["channels"], 2)
        audio = standardize_audio(path)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.size, 16000)
        self.assertAlmostEqual(float(np.mean(audio)), 0.05, places=2)

    def test_resamples_to_16khz(self):
        path = self.wav("8k.wav", np.full(8000, 0.1, dtype=np.float32), 8000)
        self.assertEqual(standardize_audio(path, 16000).size, 16000)

    def test_empty_audio_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_audio(np.array([], dtype=np.float32), 16000)

    def test_nan_audio_raises(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_audio(np.array([0.1, np.nan], dtype=np.float32), 16000)

    def test_silence_rejected_and_near_silence_flagged(self):
        with self.assertRaisesRegex(ValueError, "silent"):
            validate_audio(np.zeros(16000, dtype=np.float32), 16000)
        metrics = compute_quality_metrics(np.full(16000, 1e-4, dtype=np.float32), 16000)
        self.assertTrue(metrics["low_signal"])
        self.assertFalse(compute_quality_metrics(np.full(16000, 0.1), 16000)["low_signal"])

    def test_60_second_recording_is_one_partial_window(self):
        windows = split_into_windows(np.ones(60, dtype=np.float32), sr=1)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].size, 60)

    def test_more_than_20_minutes_splits_and_keeps_partial(self):
        audio = np.ones(1200 * 2 + 17, dtype=np.float32)
        windows = split_into_windows(audio, sr=1)
        self.assertEqual([len(w) for w in windows], [1200, 1200, 17])

    def test_manifest_offsets_fields_and_files(self):
        windows = [np.full(10, 0.1, dtype=np.float32), np.full(4, 0.2, dtype=np.float32)]
        out = self.root / "prepared" / "session"
        records = build_window_manifest("session", windows, 2, out)
        self.assertEqual(set(records[0]), set(MANIFEST_FIELDS))
        self.assertEqual([r["window_id"] for r in records], [0, 1])
        self.assertEqual([r["audio_path"] for r in records], [
            str(out / "window_000.wav"), str(out / "window_001.wav")])
        self.assertEqual((records[0]["start_offset_s"], records[0]["end_offset_s"]), (0.0, 5.0))
        self.assertEqual((records[1]["start_offset_s"], records[1]["end_offset_s"]), (5.0, 7.0))
        for record in records:
            info = sf.info(record["audio_path"])
            self.assertEqual(info.format, "WAV")
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(info.channels, 1)

    def test_manifest_writer_emits_parseable_metadata(self):
        out = self.root / "windows"
        records = build_window_manifest("sid", [np.ones(8, dtype=np.float32)], 4, out)
        self.assertEqual(records[0]["duration_s"], 2.0)
        self.assertEqual(records[0]["status"], "ready")

    def test_save_window_pcm16(self):
        path = save_window(np.full(100, 0.25, dtype=np.float32), 16000, self.root / "x.wav")
        self.assertTrue(sf.info(path).samplerate == 16000)
        self.assertEqual(sf.info(path).subtype, "PCM_16")


if __name__ == "__main__":
    unittest.main()
