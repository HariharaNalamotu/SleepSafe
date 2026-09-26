"""Replay a recording through the same path the Pi uses: 10 s chunks -> volume -> job -> report.

    set DATABRICKS_HOST=... & set DATABRICKS_TOKEN=...
    python scripts/simulate_pi.py recording.wav --session-id demo-night-1 --job-id <id> [--demo-mode] [--wait]
    python scripts/simulate_pi.py chunks_dir/  --session-id ...           (already chunked)

Uploads chunks/<session_id>/<index>.flac + session.json to the volume, triggers the processing job
and (with --wait) polls until it finishes, then prints the summary from the Delta tables.
"""
import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
import soundfile as sf

HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
HDR = {"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}


def put(path: str, data: bytes):
    for i in range(5):
        r = requests.put(f"{HOST}/api/2.0/fs/files{path}?overwrite=true", data=data, timeout=60,
                         headers={**HDR, "Content-Type": "application/octet-stream"})
        if r.ok:
            return
        time.sleep(2 * (i + 1))
    r.raise_for_status()


def chunks_from(src: Path, chunk_s: int):
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in (".flac", ".wav"):
                yield p.stem, p.read_bytes()
        return
    x, sr = sf.read(str(src), dtype="int16", always_2d=True)
    x = x.mean(1).astype("int16")
    n = chunk_s * sr
    for i in range(0, len(x), n):
        buf = io.BytesIO()
        sf.write(buf, x[i:i + n], sr, format="FLAC")
        yield f"{i // n:05d}", buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--volume", default="/Volumes/workspace/default/sleepsafe")
    ap.add_argument("--start-utc", default=None)
    ap.add_argument("--chunk-s", type=int, default=10)
    ap.add_argument("--demo-mode", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--warehouse-id", default=None, help="to print results from Delta after --wait")
    args = ap.parse_args()

    base = f"{args.volume}/chunks/{args.session_id}"
    start = args.start_utc or datetime.now(timezone.utc).isoformat()
    put(f"{base}/session.json", json.dumps({"start_utc": start}).encode())
    items = list(chunks_from(Path(args.input), args.chunk_s))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda kv: put(f"{base}/{kv[0]}.flac", kv[1]), items))
    print(f"uploaded {len(items)} chunks to {base} in {time.time() - t0:.0f}s")

    r = requests.post(f"{HOST}/api/2.2/jobs/run-now", headers=HDR, timeout=30, json={
        "job_id": args.job_id,
        "job_parameters": {"session_id": args.session_id, "demo_mode": str(args.demo_mode).lower()}})
    r.raise_for_status()
    run_id = r.json()["run_id"]
    print(f"processing run {run_id}: {HOST}/jobs/{args.job_id}/runs/{run_id}")
    if not args.wait:
        return
    while True:
        run = requests.get(f"{HOST}/api/2.2/jobs/runs/get", headers=HDR, params={"run_id": run_id}, timeout=30).json()
        st = run["state"]
        if st["life_cycle_state"] in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            print(f"run finished: {st.get('result_state')} {st.get('state_message', '')}")
            break
        time.sleep(15)
    if args.warehouse_id and st.get("result_state") == "SUCCESS":
        q = requests.post(f"{HOST}/api/2.0/sql/statements", headers=HDR, timeout=60, json={
            "warehouse_id": args.warehouse_id, "wait_timeout": "50s",
            "statement": "SELECT * FROM workspace.default.sleepsafe_v_sessions WHERE session_id = :sid",
            "parameters": [{"name": "sid", "value": args.session_id}]}).json()
        cols = [c["name"] for c in q["manifest"]["schema"]["columns"]]
        for row in q["result"].get("data_array", []):
            print(json.dumps(dict(zip(cols, row)), indent=1))


if __name__ == "__main__":
    main()
