"""Databricks job: process one finished session end-to-end (serverless CPU).

Job parameters (Python script task, passed as --key value):
    --session_id     required
    --volume_root    /Volumes/workspace/default/sleepsafe  (holds chunks/, raw/, models/, reports/)
    --catalog, --schema     where the Delta tables live (default workspace.default)
    --llm_endpoint   Foundation Model API chat endpoint used for the written report
    --demo_mode      "true" to count events regardless of sleep/wake
    --delete_chunks  "true" to delete raw audio after processing

Audio input (first match wins):
    chunks/<session_id>/<index>.flac|wav    10 s chunks uploaded by the Pi (+ optional session.json)
    raw/<session_id>/recording.wav          one full recording

Outputs (the tables the sleepsafe-agent app reads, plus the written report):
    <catalog>.<schema>.sleepsafe_sessions   session_id, report_json            (one row per session)
    <catalog>.<schema>.sleepsafe_events     session_id, start_offset_s, event_json
    <catalog>.<schema>.sleepsafe_reports    session_id, report_markdown, llm_endpoint, created_utc
    <volume_root>/reports/<session_id>.json / .md
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
    ap.add_argument("--volume_root", default="/Volumes/workspace/default/sleepsafe")
    ap.add_argument("--catalog", default="workspace")
    ap.add_argument("--schema", default="default")
    ap.add_argument("--llm_endpoint", default="databricks-gpt-oss-120b")
    ap.add_argument("--demo_mode", default="false")
    ap.add_argument("--delete_chunks", default="false")
    return ap.parse_args()


REPORT_SYSTEM_PROMPT = """You are a sleep-health screening assistant. You receive the JSON output of an
audio-based sleep monitoring model (lapel microphone) for one session and write a clear report for the
user who wore the microphone.

Rules:
- This is a screening aid, NOT a diagnosis. Never state or imply that the user has a condition: do not name
  disorders or syndromes (e.g. "obstructive sleep apnea syndrome", COPD), do not say how a clinical study
  "might classify" the findings, and do not label events obstructive or central — audio cannot tell.
  Describe what was detected ("breathing pauses", "reduced breathing") and recommend a clinical sleep study
  (polysomnography) when findings warrant it.
- Base every statement on the JSON. Quote times in UTC (e.g. "around 02:51 UTC") and counts from it.
- Weigh events by `confidence` and `confidence_basis`: "trained" events (apnea, hypopnea, snore_episode,
  wake_period) are more reliable than "zero_shot" ones (cough, gasp, snort, wheeze, speech).
- Respect every item in `caveats`, especially short-session and demo-mode caveats; mention them plainly.
- Mention `low_signal` periods as times the audio could not be assessed.

Format (Markdown):
# Sleep Screening Report
**Session:** … | **Recorded:** start–end (UTC) | **Duration:** …
## Summary            (3–5 sentences, plain language)
## Key Findings       (bullets: breathing pauses, snoring, other sounds, with numbers)
## Timeline           (table of the most notable events: time, type, duration, confidence)
## What This Could Mean  (careful, non-diagnostic)
## Recommendations    (practical next steps; when to see a doctor)
## Limitations        (from caveats)
"""


def message_text(content) -> str:
    """Chat content can be a string or a list of typed parts (e.g. reasoning + text)."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        p = p if isinstance(p, dict) else getattr(p, "__dict__", {})
        if p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def main():
    args = parse_args()
    root = Path(args.volume_root)
    try:  # code checked out in the workspace: <repo>/databricks/process_session.py
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    except NameError:
        sys.path.insert(0, "/Workspace/Users/yahuja2@wisc.edu/sleepsafe-model")
    from pyspark.sql import SparkSession

    from sleepsafe.pipeline import SleepSafePipeline, assemble_chunks, load_audio

    spark = SparkSession.builder.getOrCreate()
    t0 = time.time()
    sid = args.session_id
    chunk_dir = root / "chunks" / sid
    raw_file = root / "raw" / sid / "recording.wav"
    meta = {}
    files = []
    if chunk_dir.exists():
        meta_path = chunk_dir / "session.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        files = [p for p in chunk_dir.iterdir() if p.suffix.lower() in (".flac", ".wav")]
    if files:
        print(f"{len(files)} chunks for {sid}", flush=True)
        audio, valid = assemble_chunks(files)
    elif raw_file.exists():
        print(f"single recording {raw_file}", flush=True)
        audio, valid = load_audio(raw_file), None
    else:
        raise FileNotFoundError(f"no audio for session {sid} in {chunk_dir} or {raw_file}")

    start = datetime.fromisoformat(meta["start_utc"]) if "start_utc" in meta else datetime.now(timezone.utc)
    pipe = SleepSafePipeline(root / "models" / "Cnn14_16k.pth", root / "models" / "model.pt", device="cpu")
    night = pipe.run(audio, sid, start, valid, demo_mode=args.demo_mode.lower() == "true")
    print(f"model done in {time.time() - t0:.0f}s: {json.dumps(night['summary'])}", flush=True)

    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sid}.json").write_text(json.dumps(night, indent=2))

    # ---- Delta tables (schema used by the sleepsafe-agent app) ----
    t = f"{args.catalog}.{args.schema}"
    spark.sql(f"CREATE TABLE IF NOT EXISTS {t}.sleepsafe_sessions (session_id STRING, report_json STRING) USING DELTA")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {t}.sleepsafe_events "
              "(session_id STRING, start_offset_s DOUBLE, event_json STRING) USING DELTA")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {t}.sleepsafe_reports "
              "(session_id STRING, report_markdown STRING, llm_endpoint STRING, created_utc STRING) USING DELTA")
    for table in ("sleepsafe_sessions", "sleepsafe_events"):  # re-processing replaces the session
        spark.sql(f"DELETE FROM {t}.{table} WHERE session_id = :sid", args={"sid": sid})
    spark.createDataFrame([(sid, json.dumps(night))], "session_id STRING, report_json STRING") \
        .write.mode("append").saveAsTable(f"{t}.sleepsafe_sessions")
    rows = [(sid, float(e["start_offset_s"]), json.dumps(e)) for e in night["events"]]
    if rows:
        spark.createDataFrame(rows, "session_id STRING, start_offset_s DOUBLE, event_json STRING") \
            .write.mode("append").saveAsTable(f"{t}.sleepsafe_events")
    print(f"wrote {len(rows)} events", flush=True)

    # ---- written report (Foundation Model API) ----
    report_md = None
    try:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        resp = client.chat.completions.create(
            model=args.llm_endpoint,
            messages=[{"role": "system", "content": REPORT_SYSTEM_PROMPT},
                      {"role": "user", "content": json.dumps(night)}],
            max_tokens=4000,
        )
        report_md = message_text(resp.choices[0].message.content).strip()
    except Exception as e:  # the model output is already stored; the report can be regenerated
        print(f"report generation failed: {e}", flush=True)
    if report_md:
        (out_dir / f"{sid}.md").write_text(report_md)
        spark.sql(f"DELETE FROM {t}.sleepsafe_reports WHERE session_id = :sid", args={"sid": sid})
        spark.createDataFrame(
            [(sid, report_md, args.llm_endpoint, datetime.now(timezone.utc).isoformat())],
            "session_id STRING, report_markdown STRING, llm_endpoint STRING, created_utc STRING",
        ).write.mode("append").saveAsTable(f"{t}.sleepsafe_reports")
        print(f"report written ({len(report_md)} chars)", flush=True)

    if args.delete_chunks.lower() == "true":
        for p in files:
            p.unlink()
    print(f"session {sid} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
