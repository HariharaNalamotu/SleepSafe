# SleepSafe agent layer

This folder is the complete, runnable hackathon project. The CRNN produces a report;
this app reads it, classifies its respiratory *pattern*, and answers questions. It does
not diagnose disease. The sample report is copied verbatim from the brief.

## Files

| File | Purpose |
|---|---|
| `app.py`, `app.yaml` | FastAPI Databricks App and launch command |
| `requirements.txt` | App dependencies |
| `sleepsafe/system_prompt.txt` | Agent policy |
| `sleepsafe/data/sample_report.json` | Exact mock CRNN output |
| `sleepsafe/repository.py` | Sample and Delta data sources |
| `sleepsafe/tools.py` | Four report-reading tools |
| `sleepsafe/classifier.py` | Deterministic classification |
| `sleepsafe/agent.py` | Question answering and optional tool-calling LLM |
| `sleepsafe/schema.py` | Input/output validation |
| `demo.py`, `tests/test_sleepsafe.py` | Prompt harness and eight tests |

## Local test

From this folder:

```bash
python3 -m unittest discover -s tests -v
python3 demo.py
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

## Pre-ML audio preprocessing

The independent `preprocessing/` package standardizes WAV files to 16 kHz mono,
checks basic usability and signal quality, writes 20-minute PCM-16 windows
(including a final partial window), and produces a local `manifest.json`. It
does not diagnose, generate MFCCs or Mel spectrograms, or alter the post-ML
agent. Install dependencies with `pip install -r requirements.txt`.

Local example:

```bash
python scripts/preprocess_audio.py \
  --input /path/to/recording.wav \
  --session-id test-session-001 \
  --output-dir /path/to/prepared/test-session-001
```

For Databricks, place the raw file at
`/Volumes/workspace/default/sleepsafe/raw/<session_id>/recording.wav`, then run
the manual serverless job **SleepSafe audio preprocessing** (job ID
`641484757512224`). This job is already configured with the preprocessing
package, dependencies, Volume paths, and `--write-delta`.

```bash
databricks fs cp /path/to/recording.wav \
  dbfs:/Volumes/workspace/default/sleepsafe/raw/test-session-001/recording.wav \
  --profile sleepsafe
databricks jobs run-now 641484757512224 --profile sleepsafe \
  --json '{"job_parameters":{"session_id":"test-session-001"}}'
```

The prepared WAVs go to
`/Volumes/workspace/default/sleepsafe/prepared/test-session-001/`; the JSON
manifest is `manifest.json` in that directory. The job appends window metadata
to `workspace.default.sleepsafe_audio_windows`. It uses the session ID to build
the raw and prepared paths. For the metadata schema and CRNN interface, see
`CONTEXT.md`. The `low_signal` flag is an audio quality heuristic, not a
clinical threshold. The job has not been run yet; upload an actual recording
before starting it.

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/classify/a3f9c2e1-lapel-0926
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"session_id":"a3f9c2e1-lapel-0926","question":"What happened around 12:18 AM?"}'
```

`/classify` always returns exactly seven fields. It preserves `severity_estimate`.
The 70% valid-audio threshold, AHI 5 boundary, and 30% snoring threshold are
hackathon classification heuristics, not clinical cutoffs. The sample classifies
as `mixed_respiratory_pattern`. Its summary counts and listed events disagree;
the response reports this discrepancy instead of inventing event details.

## Deploy to Databricks Apps

1. Install a current Databricks CLI and authenticate with your workspace:

   ```bash
   databricks auth login --host https://YOUR-WORKSPACE-HOST
   databricks current-user me
   ```

2. Create a custom App named `sleepsafe-agent` in **Databricks Apps** in the
   workspace UI. The workspace must support Databricks Apps/serverless compute.
   For the sample data, no model endpoint or SQL warehouse is required.

3. From this folder, upload and deploy:

   ```bash
   databricks sync . /Workspace/Users/YOUR-EMAIL/sleepsafe-agent
   databricks apps deploy sleepsafe-agent \
     --source-code-path /Workspace/Users/YOUR-EMAIL/sleepsafe-agent
   databricks apps get sleepsafe-agent
   ```

4. Open the App URL shown by `apps get`. Visit `/docs` for Swagger UI. Use
   `/classify/a3f9c2e1-lapel-0926` and `POST /ask` with the JSON above. If
   deployment fails, check the App's **Logs** tab. Redeploy with the same sync
   and deploy commands after edits.

## Optional Databricks LLM for open-ended questions

The nine demo questions have deterministic answers. For other questions, set
`SLEEPSAFE_LLM_ENDPOINT` to a chat-completions serving endpoint that supports
function tools. In the Apps UI, add that serving endpoint as an App resource
with **CAN QUERY** permission for the App service principal. Add this to
`app.yaml` (replace the endpoint name):

```yaml
env:
  - name: SLEEPSAFE_LLM_ENDPOINT
    value: YOUR-TOOL-CALLING-ENDPOINT
```

The LLM receives a system prompt and tool definitions, calls tools as needed,
and cannot override the deterministic `/classify` result. This route has not
been live-tested without a workspace endpoint. If the endpoint does not
support tool calling, use the deterministic question set until you select one.

## Connect real CRNN reports to Delta

Create two Unity Catalog Delta tables. `report_json` contains the full CRNN
JSON; `event_json` contains each event. The repository uses parameterized SQL.

```sql
CREATE TABLE IF NOT EXISTS workspace.default.sleepsafe_sessions (
  session_id STRING, report_json STRING
) USING DELTA;
CREATE TABLE IF NOT EXISTS workspace.default.sleepsafe_events (
  session_id STRING, start_offset_s DOUBLE, event_json STRING
) USING DELTA;
```

Your CRNN pipeline writes one `report_json` row per session and one
`event_json` row per event. Store the original full report including summary,
recording, and caveats. A notebook ingestion example:

```python
import json
report = YOUR_CRNN_REPORT_DICT
sid = report["session_id"]
spark.createDataFrame([(sid, json.dumps(report))],
    "session_id STRING, report_json STRING").write.mode("append").saveAsTable(
    "workspace.default.sleepsafe_sessions")
rows = [(sid, float(e["start_offset_s"]), json.dumps(e))
        for e in report.get("events", [])]
if rows:
    spark.createDataFrame(rows,
        "session_id STRING, start_offset_s DOUBLE, event_json STRING").write.mode(
        "append").saveAsTable("workspace.default.sleepsafe_events")
```

Before switching, grant the App service principal **CAN USE** on a SQL warehouse
and Unity Catalog **USE CATALOG**, **USE SCHEMA**, and **SELECT** on both tables.
Find the warehouse HTTP path in its connection details. Add these entries to
`app.yaml` and redeploy:

```yaml
env:
  - name: SLEEPSAFE_REPOSITORY
    value: delta
  - name: DATABRICKS_HTTP_PATH
    value: /sql/1.0/warehouses/YOUR-WAREHOUSE-ID
```

If also using an LLM, combine all three entries under one `env:` list.
`get_session_from_delta` and `get_events_from_delta` in `repository.py` are
already implemented. Avoid duplicate `session_id` rows when your pipeline
retries; use `MERGE` for production ingestion.

## Connect a frontend

The frontend sends `GET /classify/{session_id}` for the seven-field card and
`POST /ask` for chat, with body `{"session_id":"...","question":"..."}`.
For another Databricks App, grant its service principal **CAN USE** on this App
and call the SleepSafe App URL with Databricks OAuth authorization. For a browser
UI, host the frontend in this same App and use relative API paths so the App's
sign-in covers both UI and API. Avoid exposing raw report JSON by default.

## Limits

`get_event_near_time` interprets a question such as `12:18 AM` in UTC because
the report has UTC timestamps and no local timezone. Add the user's timezone
to the request before supporting local-time questions. A previous-night query
needs another report in the repository. The model-reported longest apnea may
not have a matching event in the supplied list, so the answer says so.
