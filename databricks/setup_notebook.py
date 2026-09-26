# Databricks notebook source
# MAGIC %md
# MAGIC # SleepSafe — one-time workspace setup
# MAGIC Run all cells. Idempotent: safe to re-run.
# MAGIC 1. Schema, volume, Delta tables and dashboard views
# MAGIC 2. Model files into the volume (v2 head from this repo, CNN14 encoder downloaded)
# MAGIC 3. LLM endpoint for the report agent
# MAGIC 4. The `sleepsafe-process-session` job

# COMMAND ----------

import json
import os
import shutil

CATALOG, SCHEMA, VOLUME = "workspace", "sleepsafe", "data"
T = f"{CATALOG}.{SCHEMA}"
ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
NB_PATH = ctx.notebookPath().get()                      # /Users/<me>/SleepSafe/databricks/setup_notebook
REPO = "/Workspace" + NB_PATH.rsplit("/databricks/", 1)[0]
print("repo:", REPO)

# COMMAND ----------

# 1. Catalog objects
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {T}")  # noqa: F821
spark.sql(f"CREATE VOLUME IF NOT EXISTS {T}.{VOLUME}")  # noqa: F821
for d in ("models", "chunks", "reports"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)

spark.sql(f"""CREATE TABLE IF NOT EXISTS {T}.sessions (
  session_id STRING, start_utc STRING, end_utc STRING, duration_s BIGINT, valid_audio_s BIGINT,
  estimated_sleep_s BIGINT, events_per_hour DOUBLE, severity_estimate STRING, snore_pct_of_sleep DOUBLE,
  counts_json STRING, night_json STRING, processed_utc STRING)""")  # noqa: F821
spark.sql(f"""CREATE TABLE IF NOT EXISTS {T}.incidents (
  session_id STRING, event_id STRING, type STRING, start_offset_s BIGINT, end_offset_s BIGINT,
  duration_s BIGINT, start_utc STRING, confidence DOUBLE, confidence_basis STRING)""")  # noqa: F821
spark.sql(f"""CREATE TABLE IF NOT EXISTS {T}.reports (
  session_id STRING, report_markdown STRING, llm_endpoint STRING, created_utc STRING)""")  # noqa: F821

# Dashboard views: one row per session (latest processing wins)
spark.sql(f"""CREATE OR REPLACE VIEW {T}.v_sessions AS
SELECT * EXCEPT (rn) FROM (
  SELECT *, row_number() OVER (PARTITION BY session_id ORDER BY processed_utc DESC) rn FROM {T}.sessions)
WHERE rn = 1""")  # noqa: F821
spark.sql(f"""CREATE OR REPLACE VIEW {T}.v_reports AS
SELECT * EXCEPT (rn) FROM (
  SELECT *, row_number() OVER (PARTITION BY session_id ORDER BY created_utc DESC) rn FROM {T}.reports)
WHERE rn = 1""")  # noqa: F821
spark.sql(f"""CREATE OR REPLACE VIEW {T}.v_incidents AS
SELECT i.* FROM {T}.incidents i
JOIN (SELECT session_id, max(processed_utc) p FROM {T}.sessions GROUP BY session_id) s USING (session_id)""")  # noqa: F821
spark.sql(f"""CREATE OR REPLACE VIEW {T}.v_event_counts AS
SELECT session_id, type, count(*) AS n, round(sum(duration_s) / 60, 1) AS total_min,
       round(avg(confidence), 3) AS avg_confidence
FROM {T}.v_incidents GROUP BY session_id, type""")  # noqa: F821
print("tables + views ready")

# COMMAND ----------

# 2. Model files
shutil.copy(f"{REPO}/runs/v2/model.pt", f"{ROOT}/models/model.pt")
enc = f"{ROOT}/models/Cnn14_16k.pth"
ENC_SIZE = 358668570
if not (os.path.exists(enc) and os.path.getsize(enc) == ENC_SIZE):
    import requests
    url = "https://huggingface.co/niobures/PANNs/resolve/main/models/pretrained/Cnn14_16k_mAP%3D0.438.pth"
    try:
        tmp = "/tmp/Cnn14_16k.pth"
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk)
        assert os.path.getsize(tmp) == ENC_SIZE, os.path.getsize(tmp)
        shutil.copy(tmp, enc)
        print("encoder downloaded")
    except Exception as e:  # outbound internet can be restricted on Free Edition
        print(f"!! could not download encoder ({e}). Upload data/pretrained/Cnn14_16k.pth to {ROOT}/models/ manually.")
print({f: os.path.getsize(f"{ROOT}/models/{f}") for f in os.listdir(f"{ROOT}/models")})

# COMMAND ----------

# 3. LLM endpoint for the report agent (prefer Claude)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
names = [e.name for e in w.serving_endpoints.list()]
print("serving endpoints:", names)
prefs = ["claude-sonnet", "claude-opus", "claude", "gpt-oss-120b", "llama-4", "llama-3-3-70b", "gpt-oss", "llama"]
LLM = next((n for p in prefs for n in names if p in n), None)
assert LLM, "no chat endpoint found"
print("report agent LLM:", LLM)

# COMMAND ----------

# 4. Processing job (serverless CPU)
job_name = "sleepsafe-process-session"
spec = {
    "name": job_name,
    "max_concurrent_runs": 2,
    "parameters": [{"name": "session_id", "default": ""}, {"name": "demo_mode", "default": "false"}],
    "tasks": [{
        "task_key": "process_session",
        "environment_key": "sleepsafe_env",
        "spark_python_task": {
            "python_file": f"{REPO}/databricks/process_session.py",
            "parameters": ["--session_id", "{{job.parameters.session_id}}",
                           "--demo_mode", "{{job.parameters.demo_mode}}",
                           "--volume_root", ROOT, "--catalog", CATALOG, "--schema", SCHEMA,
                           "--llm_endpoint", LLM, "--delete_chunks", "false"],
        },
        "timeout_seconds": 1800,
    }],
    "environments": [{
        "environment_key": "sleepsafe_env",
        "spec": {"client": "4", "dependencies": [
            "--extra-index-url https://download.pytorch.org/whl/cpu",
            "torch==2.11.0+cpu", "numpy", "scipy", "soundfile", "openai", "databricks-sdk"]},
    }],
}
existing = [j.job_id for j in w.jobs.list(name=job_name)]
if existing:
    w.api_client.do("POST", "/api/2.2/jobs/reset", body={"job_id": existing[0], "new_settings": spec})
    JOB_ID = existing[0]
else:
    JOB_ID = w.api_client.do("POST", "/api/2.2/jobs/create", body=spec)["job_id"]
print("JOB_ID =", JOB_ID)

# COMMAND ----------

# 5. Summary for the Pi uploader and the dashboard
wh = [x for x in w.warehouses.list()]
print(json.dumps({
    "host": w.config.host,
    "volume": ROOT,
    "job_id": JOB_ID,
    "llm_endpoint": LLM,
    "sql_warehouse_id": wh[0].id if wh else None,
    "tables": [f"{T}.v_sessions", f"{T}.v_incidents", f"{T}.v_event_counts", f"{T}.v_reports"],
}, indent=2))
