"""Raspberry Pi: upload 10 s audio chunks to a Databricks volume, then trigger processing.

Your capture code writes chunks into --watch-dir as <index>.flac (or .wav), 16 kHz mono.
Keep the capture close to raw: NO noise suppression, noise gate or automatic gain control —
they erase quiet breathing and fake/hide breathing pauses. Resampling and a ~50 Hz high-pass are fine.

    export DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
    export DATABRICKS_TOKEN=<personal access token>
    python uploader.py --watch-dir /tmp/chunks --volume /Volumes/workspace/default/sleepsafe \
        --job-id 123456789 [--duration-min 30] [--demo-mode]

Stop with Ctrl+C (or --duration-min): remaining chunks are flushed, then the job is triggered.
"""
import argparse
import json
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
HDR = {"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}


def put_file(volume_path: str, data: bytes, retries: int = 5) -> None:
    url = f"{HOST}/api/2.0/fs/files{volume_path}?overwrite=true"
    for i in range(retries):
        try:
            r = requests.put(url, headers={**HDR, "Content-Type": "application/octet-stream"},
                             data=data, timeout=30)
            if r.ok:
                return
            print(f"upload {volume_path}: HTTP {r.status_code} {r.text[:200]}")
        except requests.RequestException as e:
            print(f"upload {volume_path}: {e}")
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"giving up on {volume_path}")


def run_job(job_id: int, params: dict) -> int:
    r = requests.post(f"{HOST}/api/2.2/jobs/run-now", headers=HDR,
                      json={"job_id": job_id, "job_parameters": params}, timeout=30)
    r.raise_for_status()
    return r.json()["run_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", required=True)
    ap.add_argument("--volume", required=True, help="/Volumes/<catalog>/<schema>/<volume>")
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--duration-min", type=float, default=None)
    ap.add_argument("--demo-mode", action="store_true")
    ap.add_argument("--delete-local", action="store_true")
    args = ap.parse_args()

    session = args.session_id or f"{datetime.now():%Y%m%d-%H%M}-{uuid.uuid4().hex[:6]}"
    base = f"{args.volume}/chunks/{session}"
    start = datetime.now(timezone.utc)
    put_file(f"{base}/session.json", json.dumps({"start_utc": start.isoformat()}).encode())
    print(f"session {session} -> {base}")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    watch = Path(args.watch_dir)
    done: set[str] = set()
    deadline = time.time() + args.duration_min * 60 if args.duration_min else None

    def flush(final: bool):
        files = sorted((p for p in watch.iterdir() if p.suffix.lower() in (".flac", ".wav")),
                       key=lambda p: p.stat().st_mtime)
        if not final and files:
            files = files[:-1]  # newest chunk may still be being written
        for p in files:
            if p.name in done:
                continue
            put_file(f"{base}/{p.name}", p.read_bytes())
            done.add(p.name)
            if args.delete_local:
                p.unlink()
            print(f"uploaded {p.name} ({len(done)} total)")

    while not stop["flag"] and (deadline is None or time.time() < deadline):
        flush(final=False)
        time.sleep(2)
    time.sleep(1)
    flush(final=True)
    run_id = run_job(args.job_id, {"session_id": session, "demo_mode": str(args.demo_mode).lower()})
    print(f"session {session}: {len(done)} chunks uploaded; processing run {run_id} started")


if __name__ == "__main__":
    main()
