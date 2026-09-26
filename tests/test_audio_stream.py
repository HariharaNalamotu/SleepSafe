from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
import wave

import audio_stream as stream


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.spool = Path(self.temp.name)

    def test_exact_wav_and_atomic_visibility(self):
        pcm = b"\x01\x00" * (16000 * 10)
        path = stream.save_chunk(self.spool, pcm, "pi-01")
        with wave.open(str(path), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                              wav.getnframes()), (16000, 1, 2, 160000))
            self.assertEqual(wav.readframes(160000), pcm)
        self.assertFalse(list(self.spool.glob("*.part")))
        self.assertFalse(stream.spool_has_room(self.spool, stream.CHUNK_BYTES + 44))
        with self.assertRaises(ValueError):
            stream.save_chunk(self.spool, b"\0\0", "pi-01")

    def test_failed_upload_retains_file_and_success_removes_it(self):
        path = stream.save_chunk(self.spool, bytes(stream.CHUNK_BYTES), "pi-01")
        uploader = Mock()
        uploader.upload.side_effect = URLError("offline")
        with self.assertRaises(URLError):
            stream.upload_one(path, uploader)
        self.assertTrue(path.exists())
        uploader.upload.side_effect = None
        stream.upload_one(path, uploader)
        self.assertFalse(path.exists())

    def test_files_api_request_and_stable_retry_path(self):
        path = stream.save_chunk(self.spool, bytes(stream.CHUNK_BYTES), "pi-01")
        client = stream.DatabricksUploader("https://example.databricks.com", "test-token",
                                          "/Volumes/catalog/schema/volume/audio folder")
        response = Mock(status=204)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client.opener = Mock()
        client.opener.open.return_value = response
        client.upload(path)
        client.upload(path)
        calls = client.opener.open.call_args_list
        self.assertEqual(len(calls), 4)
        request = calls[1].args[0]
        self.assertIn("/api/2.0/fs/files/Volumes/catalog/schema/volume/audio%20folder/", request.full_url)
        self.assertEqual(request.full_url, calls[3].args[0].full_url)
        self.assertTrue(request.full_url.endswith("?overwrite=true"))
        self.assertEqual(request.method, "PUT")
        self.assertEqual(request.data, path.read_bytes())
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")

    def test_reject_invalid_cloud_configuration(self):
        for host in ("http://example.com", "https://example.com/path", "https://a:b@example.com"):
            with self.assertRaises(ValueError):
                stream.DatabricksUploader(host, "token", "/Volumes/c/s/v")
        for volume in ("/tmp/audio", "/Volumes/c/s", "/Volumes/c/s/v/../x"):
            with self.assertRaises(ValueError):
                stream.DatabricksUploader("https://example.com", "token", volume)

    def test_retry_backoff_and_queue_survives_failure(self):
        path = stream.save_chunk(self.spool, bytes(stream.CHUNK_BYTES), "pi-01")
        uploader = Mock()
        uploader.upload.side_effect = HTTPError("https://example.com", 503, "busy", {}, None)
        stop = Mock()
        stop.is_set.side_effect = [False, False, False, True]
        stream.upload_loop(self.spool, uploader, stop)
        self.assertEqual([c.args[0] for c in stop.wait.call_args_list], [1, 2, 4])
        self.assertTrue(path.exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_real_ffmpeg_resample_and_capture_chunk_boundaries(self):
        # 48 kHz stereo fixture, 20.25 seconds -> two full chunks, partial tail.
        command = stream.ffmpeg_command("unused", "highpass=f=60,lowpass=f=7000")
        start = command.index("-f")
        command[start:start + 4] = [
            "-f", "lavfi", "-i",
            "aevalsrc=0.1*sin(2*PI*440*t)|0.1*sin(2*PI*880*t):s=48000:d=20.25",
        ]
        args = Mock(source="unused", filters="highpass=f=60,lowpass=f=7000",
                    spool=self.spool, max_spool_mb=10, device_id="pi-test")
        with patch.object(stream, "ffmpeg_command", return_value=command):
            with self.assertRaisesRegex(OSError, "stream ended"):
                stream.capture_once(args, threading.Event())
        files = sorted(self.spool.glob("*.wav"))
        self.assertEqual(len(files), 2)
        for path in files:
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getnframes(), 160000)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertNotEqual(wav.readframes(160000), bytes(stream.CHUNK_BYTES))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_narrowband_input_resamples_before_filtering(self):
        command = stream.ffmpeg_command("unused", "highpass=f=60,lowpass=f=7000")
        start = command.index("-f")
        command[start:start + 4] = ["-f", "lavfi", "-i", "sine=sample_rate=8000:duration=10"]
        result = subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(len(result.stdout), stream.CHUNK_BYTES)
        self.assertEqual(result.stderr, b"")


if __name__ == "__main__":
    unittest.main()
