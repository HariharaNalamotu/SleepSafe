"""Deploy SleepSafe to a Databricks workspace over the REST API (idempotent).

    set DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
    set DATABRICKS_TOKEN=<personal access token>
    python scripts/deploy_databricks.py

Creates: schema + volume + tables + dashboard views, uploads the model files, imports the code
into the workspace, picks an LLM endpoint and creates/updates the processing job.
Writes the resulting ids to scratch/deploy_config.json.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
S = requests.Session()
S.headers["Authorization"] = f"Bearer {os.environ['DATABRICKS_TOKEN']}"

CATALOG, SCHEMA, VOLUME = "workspace", "default", "sleepsafe"
T = f"{CATALOG}.{SCHEMA}"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
JOB_NAME = "sleepsafe-process-session"
CODE_FILES = [
    "sleepsafe/__init__.py", "sleepsafe/encoder.py", "sleepsafe/model.py", "sleepsafe/labels.py",
    "sleepsafe/postprocess.py", "sleepsafe/pipeline.py", "sleepsafe/augment.py",
    "databricks/process_session.py",
]


def api(method, path, **kw):
    r = S.request(method, HOST + path, timeout=kw.pop("timeout", 120), **kw)
    if not r.ok:
        raise RuntimeError(f"{method} {path}: HTTP {r.status_code} {r.text[:500]}")
    return r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else r


def sql(warehouse_id, statement):
    r = api("POST", "/api/2.0/sql/statements", json={
        "warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "50s",
        "on_wait_timeout": "CONTINUE"})
    while r["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(2)
        r = api("GET", f"/api/2.0/sql/statements/{r['statement_id']}")
    if r["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(f"SQL failed: {statement[:80]}... -> {r['status']}")
    return r


def upload(local: Path, remote: str):
    with open(local, "rb") as f:
        api("PUT", f"/api/2.0/fs/files{remote}?overwrite=true", data=f,
            headers={"Content-Type": "application/octet-stream"}, timeout=1800)
    print(f"  uploaded {local.name} -> {remote} ({local.stat().st_size / 1e6:.1f} MB)")


def main():
    # --- SQL warehouse ---
    whs = api("GET", "/api/2.0/sql/warehouses")["warehouses"]
    wh = whs[0]["id"]
    print(f"warehouse {whs[0]['name']} ({wh})")

    # --- catalog objects ---
    j = "get_json_object(report_json, '$.{}')"
    e = "get_json_object(event_json, '$.{}')"
    ddl = [
        f"CREATE SCHEMA IF NOT EXISTS {T}",
        f"CREATE VOLUME IF NOT EXISTS {T}.{VOLUME}",
        # tables read by the sleepsafe-agent app (written by the processing job)
        f"CREATE TABLE IF NOT EXISTS {T}.sleepsafe_sessions (session_id STRING, report_json STRING) USING DELTA",
        f"CREATE TABLE IF NOT EXISTS {T}.sleepsafe_events (session_id STRING, start_offset_s DOUBLE, event_json STRING) USING DELTA",
        f"CREATE TABLE IF NOT EXISTS {T}.sleepsafe_reports (session_id STRING, report_markdown STRING, llm_endpoint STRING, created_utc STRING) USING DELTA",
        # flat views for dashboards
        f"""CREATE OR REPLACE VIEW {T}.sleepsafe_v_sessions AS SELECT session_id,
            {j.format('recording.start_utc')} AS start_utc, {j.format('recording.end_utc')} AS end_utc,
            CAST({j.format('recording.duration_s')} AS BIGINT) AS duration_s,
            CAST({j.format('recording.valid_audio_s')} AS BIGINT) AS valid_audio_s,
            CAST({j.format('recording.estimated_sleep_s')} AS BIGINT) AS estimated_sleep_s,
            CAST({j.format('summary.estimated_ahi')} AS DOUBLE) AS estimated_ahi,
            {j.format('summary.severity_estimate')} AS severity_estimate,
            CAST({j.format('summary.counts.apnea')} AS INT) AS apnea_count,
            CAST({j.format('summary.counts.hypopnea')} AS INT) AS hypopnea_count,
            CAST({j.format('summary.snore_pct_of_sleep')} AS DOUBLE) AS snore_pct_of_sleep,
            CAST({j.format('summary.longest_apnea_s')} AS INT) AS longest_apnea_s,
            CAST({j.format('summary.short_session')} AS BOOLEAN) AS short_session,
            CAST({j.format('model.demo_mode')} AS BOOLEAN) AS demo_mode,
            {j.format('summary.counts')} AS counts_json
            FROM {T}.sleepsafe_sessions""",
        f"""CREATE OR REPLACE VIEW {T}.sleepsafe_v_events AS SELECT session_id,
            {e.format('id')} AS event_id, {e.format('type')} AS type,
            CAST({e.format('start_offset_s')} AS BIGINT) AS start_offset_s,
            CAST({e.format('end_offset_s')} AS BIGINT) AS end_offset_s,
            CAST({e.format('duration_s')} AS BIGINT) AS duration_s,
            {e.format('start_utc')} AS start_utc, {e.format('end_utc')} AS end_utc,
            CAST({e.format('confidence')} AS DOUBLE) AS confidence,
            {e.format('confidence_basis')} AS confidence_basis,
            {e.format('evidence.terminated_by')} AS terminated_by
            FROM {T}.sleepsafe_events""",
        f"""CREATE OR REPLACE VIEW {T}.sleepsafe_v_event_counts AS
            SELECT session_id, type, count(*) AS n, round(sum(duration_s) / 60, 1) AS total_min,
                   round(avg(confidence), 3) AS avg_confidence
            FROM {T}.sleepsafe_v_events GROUP BY session_id, type""",
    ]
    for s in ddl:
        sql(wh, s)
    print("schema, volume, tables, views ready")

    # --- models ---
    upload(ROOT / "runs" / "v2" / "model.pt", f"{VOL}/models/model.pt")
    enc = ROOT / "data" / "pretrained" / "Cnn14_16k.pth"
    try:
        api("HEAD", f"/api/2.0/fs/files{VOL}/models/Cnn14_16k.pth")
        print("  encoder already uploaded")
    except RuntimeError:
        upload(enc, f"{VOL}/models/Cnn14_16k.pth")

    # --- code into the workspace ---
    try:
        me = api("GET", "/api/2.0/preview/scim/v2/Me")["userName"]
    except RuntimeError:
        me = os.environ["DATABRICKS_USER"]
    base = f"/Workspace/Users/{me}/sleepsafe-model"
    for d in ("sleepsafe", "databricks"):
        api("POST", "/api/2.0/workspace/mkdirs", json={"path": f"{base}/{d}"})
    for rel in CODE_FILES:
        content = (ROOT / rel).read_bytes()
        is_nb = content.startswith(b"# Databricks notebook source")
        api("POST", "/api/2.0/workspace/import", json={
            "path": f"{base}/{rel[:-3] if is_nb else rel}", "overwrite": True,
            "format": "SOURCE" if is_nb else "AUTO", **({"language": "PYTHON"} if is_nb else {}),
            "content": base64.b64encode(content).decode()})
    print(f"code imported to {base}")

    # --- LLM endpoint for the report agent ---
    names = [e["name"] for e in api("GET", "/api/2.0/serving-endpoints").get("endpoints", [])]
    prefs = ["claude-sonnet", "claude-opus", "claude", "gpt-oss-120b", "llama-4-maverick", "llama-3-3-70b"]
    llm = next((n for p in prefs for n in names if p in n), None)
    print(f"serving endpoints: {names}\nreport agent LLM: {llm}")
    if not llm:
        sys.exit("no chat model endpoint available")

    # --- job ---
    spec = {
        "name": JOB_NAME, "max_concurrent_runs": 2,
        "parameters": [{"name": "session_id", "default": ""}, {"name": "demo_mode", "default": "false"}],
        "tasks": [{
            "task_key": "process_session", "environment_key": "sleepsafe_env", "timeout_seconds": 1800,
            "spark_python_task": {
                "python_file": f"{base}/databricks/process_session.py",
                "parameters": ["--session_id", "{{job.parameters.session_id}}",
                               "--demo_mode", "{{job.parameters.demo_mode}}",
                               "--volume_root", VOL, "--catalog", CATALOG, "--schema", SCHEMA,
                               "--llm_endpoint", llm, "--delete_chunks", "false"]},
        }],
        "environments": [{"environment_key": "sleepsafe_env", "spec": {
            "environment_version": "4", "dependencies": [
                "--extra-index-url https://download.pytorch.org/whl/cpu",
                "torch==2.11.0+cpu", "numpy", "scipy", "soundfile", "openai", "databricks-sdk"]}}],
    }
    jobs = api("GET", "/api/2.2/jobs/list", params={"name": JOB_NAME}).get("jobs", [])
    if jobs:
        job_id = jobs[0]["job_id"]
        api("POST", "/api/2.2/jobs/reset", json={"job_id": job_id, "new_settings": spec})
    else:
        job_id = api("POST", "/api/2.2/jobs/create", json=spec)["job_id"]
    print(f"job {JOB_NAME}: {job_id}")

    cfg = {"host": HOST, "volume": VOL, "job_id": job_id, "warehouse_id": wh, "llm_endpoint": llm,
           "workspace_code": base, "tables": [f"{T}.sleepsafe_v_sessions", f"{T}.sleepsafe_v_events",
                                              f"{T}.sleepsafe_v_event_counts", f"{T}.sleepsafe_reports",
                                              f"{T}.sleepsafe_sessions", f"{T}.sleepsafe_events"]}
    (ROOT / "scratch").mkdir(exist_ok=True)
    (ROOT / "scratch" / "deploy_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
