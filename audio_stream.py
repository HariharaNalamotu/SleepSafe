#!/usr/bin/env python3
"""Ubuntu Bluetooth microphone -> continuous DSP -> WAV spool -> Databricks."""

import argparse
from datetime import datetime, timezone
import fcntl
import logging
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import uuid
import wave

RATE = 16000
CHUNK_BYTES = RATE * 2 * 10
LOG = logging.getLogger("sleepsafe")


def ffmpeg_command(source, filters):
    # Filters and resampler remain alive across chunks; no per-chunk resets.
    return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-f", "pulse", "-i", source, "-vn", "-af",
            f"aformat=sample_rates={RATE}:channel_layouts=mono,{filters}",
            "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le",
            "-f", "s16le", "pipe:1"]


def save_chunk(spool, pcm, device):
    if len(pcm) != CHUNK_BYTES:
        raise ValueError("A chunk must contain exactly 10 seconds of 16 kHz mono PCM")
    # Timestamp is completion time at the Pi, not a hardware capture timestamp.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = spool / f"{stamp}_{device}_{uuid.uuid4().hex}.wav"
    pending = path.with_suffix(".part")
    with pending.open("wb") as raw:
        with wave.open(raw, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm)
        raw.flush()
        os.fsync(raw.fileno())
    pending.replace(path)  # Uploader sees only complete files.
    directory_fd = os.open(spool, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer credential to a redirected host.


class DatabricksUploader:
    def __init__(self, host, token, volume):
        parsed = urlsplit(host)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("DATABRICKS_HOST must be an HTTPS workspace origin")
        parts = volume.rstrip("/").split("/")
        if (len(parts) < 5 or parts[:2] != ["", "Volumes"]
                or any(p in ("", ".", "..") for p in parts[2:])):
            raise ValueError("DATABRICKS_VOLUME must be /Volumes/catalog/schema/volume[/folder]")
        if not token.strip():
            raise ValueError("DATABRICKS_TOKEN is required for uploads")
        self.host = host.rstrip("/")
        self.token = token
        self.volume = volume.rstrip("/")
        self.opener = build_opener(NoRedirect())

    def put(self, kind, path, data=b"", query=""):
        url = f"{self.host}/api/2.0/fs/{kind}{quote(path, safe='/')}{query}"
        request = Request(url, data=data, method="PUT", headers={
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/octet-stream",
        })
        with self.opener.open(request, timeout=30) as response:
            if not 200 <= response.status < 300:
                raise OSError(f"Unexpected HTTP status {response.status}")

    def upload(self, path):
        # UUID filename is stable on retries, including after process restart.
        directory = f"{self.volume}/{path.name[:8]}"
        self.put("directories", directory)
        self.put("files", f"{directory}/{path.name}", path.read_bytes(),
                 "?overwrite=true")


def upload_one(path, uploader):
    uploader.upload(path)
    path.unlink()  # Keep local data until the server acknowledges success.


def upload_loop(spool, uploader, stop):
    delay = 1
    while not stop.is_set():
        try:
            files = sorted(spool.glob("*.wav"))
            if not files:
                stop.wait(1)
                continue
            upload_one(files[0], uploader)
            LOG.info("Uploaded %s", files[0].name)
            delay = 1
        except (HTTPError, URLError, OSError) as exc:
            # Do not log request headers, tokens, or server response bodies.
            status = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
            LOG.warning("Upload failed (%s); local file retained; retry in %ss", status, delay)
            stop.wait(delay)
            delay = min(delay * 2, 60)


def spool_has_room(spool, max_bytes):
    used = 0
    for path in spool.iterdir():
        if path.suffix in (".wav", ".part"):
            try:
                used += path.stat().st_size
            except FileNotFoundError:
                pass  # Uploader removed a completed file during the scan.
    return used + CHUNK_BYTES + 44 <= max_bytes


def capture_once(args, stop):
    process = subprocess.Popen(ffmpeg_command(args.source, args.filters), stdout=subprocess.PIPE)
    buffer = bytearray()
    last_data = time.monotonic()
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        while not stop.is_set():
            if not selector.select(timeout=1):
                if time.monotonic() - last_data > 30:
                    raise OSError("No microphone audio for 30 seconds")
                continue
            data = os.read(process.stdout.fileno(), 65536)
            if not data:
                raise OSError("FFmpeg audio stream ended; check Bluetooth connection/profile")
            last_data = time.monotonic()
            buffer.extend(data)
            while len(buffer) >= CHUNK_BYTES:
                if not spool_has_room(args.spool, args.max_spool_mb * 1024 * 1024):
                    raise OSError("Spool limit reached; stopping capture until uploads free space")
                path = save_chunk(args.spool, bytes(buffer[:CHUNK_BYTES]), args.device_id)
                del buffer[:CHUNK_BYTES]
                LOG.info("Captured %s (10 seconds)", path.name)
    finally:
        selector.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
        if buffer:
            LOG.warning("Discarding %.3fs of uncommitted audio at capture boundary", len(buffer) / (RATE * 2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.getenv("AUDIO_SOURCE"),
                        help="Exact pactl source name (required; no default-microphone fallback)")
    parser.add_argument("--device-id", default=os.getenv("DEVICE_ID", "pi-01"))
    parser.add_argument("--spool", type=Path, default=Path(os.getenv("SPOOL_DIR", "spool")))
    parser.add_argument("--max-spool-mb", type=int, default=int(os.getenv("MAX_SPOOL_MB", "512")))
    parser.add_argument("--filters", default=os.getenv("AUDIO_FILTERS", "highpass=f=60,lowpass=f=7000"),
                        help="FFmpeg audio filter chain; use anull to bypass DSP")
    parser.add_argument("--local-only", action="store_true", help="Capture WAV files without uploading")
    args = parser.parse_args()
    if not args.source:
        parser.error("Set AUDIO_SOURCE or --source using pactl list short sources")
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", args.device_id):
        parser.error("device-id must be 1–64 letters, digits, or hyphens")
    if args.max_spool_mb < 1:
        parser.error("max-spool-mb must be positive")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Install ffmpeg first")
    os.umask(0o077)
    args.spool.mkdir(parents=True, exist_ok=True)
    # Only one producer/uploader may own a spool directory.
    lock = (args.spool / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another process already owns this spool")
    uploader = None
    if not args.local_only:
        try:
            uploader = DatabricksUploader(os.getenv("DATABRICKS_HOST", ""),
                                          os.getenv("DATABRICKS_TOKEN", ""),
                                          os.getenv("DATABRICKS_VOLUME", ""))
        except ValueError as exc:
            raise SystemExit(str(exc))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    worker = None
    if uploader:
        worker = threading.Thread(target=upload_loop, args=(args.spool, uploader, stop))
        worker.start()
    try:
        while not stop.is_set():
            if not spool_has_room(args.spool, args.max_spool_mb * 1024 * 1024):
                LOG.warning("Spool full; capture paused and audio during this interval is lost")
                stop.wait(10)
                continue
            try:
                capture_once(args, stop)
            except OSError as exc:
                LOG.warning("Capture interrupted: %s; restarting in 5s", exc)
                stop.wait(5)
    finally:
        stop.set()
        if worker:
            worker.join()  # In-flight request has a finite timeout; pending files survive.
        lock.close()


if __name__ == "__main__":
    main()
