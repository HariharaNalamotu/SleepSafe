"""Databricks job: process one finished session end-to-end (serverless CPU).

Job parameters (Python script task, passed as --key value):
    --session_id     required
    --volume_root    /Volumes/<catalog>/<schema>/<volume>   (holds chunks/, models/, reports/)
    --catalog, --schema     where the Delta tables live
    --llm_endpoint   Foundation Model API endpoint used by the report agent
    --demo_mode      "true" to count events regardless of sleep/wake
    --delete_chunks  "true" to delete raw audio after processing

Layout in the volume:
    chunks/<session_id>/<index>.flac        uploaded by the Pi every 10 s
    chunks/<session_id>/session.json        {"start_utc": ...} written by the Pi at session start
    models/Cnn14_16k.pth, models/model.pt   uploaded once
    code/sleepsafe/                         this repo's package, uploaded once
    reports/<session_id>.json / .md         outputs
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_id", required=True)
    ap.add_argument("--volume_root", required=True)
    ap.add_argument("--catalog", default="workspace")
    ap.add_argument("--schema", default="sleepsafe")
    ap.add_argument("--llm_endpoint", default="databricks-claude-sonnet-4-5")
    ap.add_argument("--demo_mode", default="false")
    ap.add_argument("--delete_chunks", default="false")
    return ap.parse_args()


REPORT_SYSTEM_PROMPT = """You are a sleep-health screening assistant. You receive the JSON output of an
audio-based sleep monitoring model (lapel microphone) for one session and write a clear report for the
user who wore the microphone.

Rules:
- This is a screening aid, NOT a diagnosis. Never state that the user has a condition. Use language such as
  "patterns consistent with" and recommend a clinical sleep study (polysomnography) when findings warrant it.
- Base every statement on the JSON. Quote times (local-time friendly, e.g. "around 00:18") and counts from it.
- Weigh events by `confidence` and `confidence_basis`: "trained" events (apnea, hypopnea, snore_episode,
  wake_period) are more reliable than "zero_shot" ones (cough, gasp, snort, wheeze, speech).
- Respect every item in `caveats`, especially short-session and demo-mode caveats; mention them plainly.
- Mention `low_signal` periods as times the audio could not be assessed.

Format (Markdown):
# Sleep Screening Report
**Session:** … | **Recorded:** start–end | **Duration:** …
## Summary            (3–5 sentences, plain language)
## Key Findings       (bullets: breathing pauses, snoring, other sounds, with numbers)
## Timeline           (table of the most notable events: time, type, duration, confidence)
## What This Could Mean  (careful, non-diagnostic)
## Recommendations    (practical next steps; when to see a doctor)
## Limitations        (from caveats)
"""


def main():
    args = parse_args()
    root = Path(args.volume_root)
    sys.path.insert(0, str(root / "code"))
    try:  # repo checked out in the workspace: <repo>/databricks/process_session.py
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    except NameError:
        pass
    import numpy as np  # noqa: F401
    from pyspark.sql import SparkSession

    from sleepsafe.pipeline import SleepSafePipeline, assemble_chunks

    spark = SparkSession.builder.getOrCreate()
    t0 = time.time()
    chunk_dir = root / "chunks" / args.session_id
    meta_path = chunk_dir / "session.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    files = [p for p in chunk_dir.iterdir() if p.suffix.lower() in (".flac", ".wav")]
    print(f"{len(files)} chunks for {args.session_id}", flush=True)
    audio, valid = assemble_chunks(files)

    start = datetime.fromisoformat(meta["start_utc"]) if "start_utc" in meta else datetime.now(timezone.utc)
    pipe = SleepSafePipeline(root / "models" / "Cnn14_16k.pth", root / "models" / "model.pt", device="cpu")
    night = pipe.run(audio, args.session_id, start, valid, demo_mode=args.demo_mode.lower() == "true")
    print(f"model done in {time.time() - t0:.0f}s: {night['summary']}", flush=True)

    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.session_id}.json").write_text(json.dumps(night, indent=2))

    # ---- Delta tables for the dashboard ----
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema}")
    tbl = f"{args.catalog}.{args.schema}"
    s, r = night["summary"], night["recording"]
    spark.createDataFrame([{
        "session_id": args.session_id, "start_utc": r["start_utc"], "end_utc": r["end_utc"],
        "duration_s": r["duration_s"], "valid_audio_s": r["valid_audio_s"],
        "estimated_sleep_s": r["estimated_sleep_s"],
        "events_per_hour": float(s["respiratory_events_per_hour"] or 0.0),
        "severity_estimate": s["severity_estimate"] or "n/a",
        "snore_pct_of_sleep": float(s["snore_pct_of_sleep"] or 0.0),
        "counts_json": json.dumps(s["counts"]), "night_json": json.dumps(night),
        "processed_utc": datetime.now(timezone.utc).isoformat(),
    }]).write.mode("append").option("mergeSchema", "true").saveAsTable(f"{tbl}.sessions")
    rows = [{
        "session_id": args.session_id, "event_id": e["id"], "type": e["type"],
        "start_offset_s": int(e["start_offset_s"]), "end_offset_s": int(e["end_offset_s"]),
        "duration_s": int(e["duration_s"]), "start_utc": e.get("start_utc"),
        "confidence": float(e.get("confidence", 0.0)), "confidence_basis": e.get("confidence_basis", "quality"),
    } for e in night["events"]]
    if rows:
        spark.createDataFrame(rows).write.mode("append").saveAsTable(f"{tbl}.incidents")

    # ---- report agent (Foundation Model API) ----
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    resp = client.chat.completions.create(
        model=args.llm_endpoint,
        messages=[{"role": "system", "content": REPORT_SYSTEM_PROMPT},
                  {"role": "user", "content": json.dumps(night)}],
        max_tokens=2500,
    )
    report_md = resp.choices[0].message.content
    (out_dir / f"{args.session_id}.md").write_text(report_md)
    spark.createDataFrame([{
        "session_id": args.session_id, "report_markdown": report_md, "llm_endpoint": args.llm_endpoint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }]).write.mode("append").saveAsTable(f"{tbl}.reports")

    if args.delete_chunks.lower() == "true":
        for p in files:
            p.unlink()
    print(f"session {args.session_id} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
