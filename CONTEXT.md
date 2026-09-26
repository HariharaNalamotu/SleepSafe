# SleepSafe project context

Last updated: 2026-09-26

## Keep this file current

Read this file before changing SleepSafe. Whenever you change code, configuration,
data, tests, deployment, or documented behavior, update the relevant section here
and add a short entry to **Change history** in the same task. Record what actually
changed and was verified; do not mark planned work as complete. Never put tokens,
passwords, client secrets, or real patient data in this file.

## Background and purpose

SleepSafe is a hackathon agent layer for CRNN sleep-audio reports. The CRNN
produces the JSON report; SleepSafe reads that report, applies deterministic
respiratory-pattern heuristics, and answers grounded questions. It does not
process raw audio, train the CRNN, or diagnose a medical condition. The bundled
report is mock data copied from the brief.

## Current state

- Local project: `/Users/yagyaahuja/Downloads/sleepsafe`.
- GitHub repository: `https://github.com/HariharaNalamotu/SleepSafe.git`;
  this project folder is its own Git checkout on branch `yagya`, tracking
  `origin/yagya`.
- Databricks workspace host: `https://dbc-0cc75cba-cde3.cloud.databricks.com`.
- Databricks CLI profile on this Mac: `sleepsafe` (interactive OAuth; no
  credentials are stored in this project).
- Workspace user/source path: `yahuja2@wisc.edu` and
  `/Workspace/Users/yahuja2@wisc.edu/sleepsafe-agent`.
- App name: `sleepsafe-agent`.
- Live App: <https://sleepsafe-agent-7474648945814205.aws.databricksapps.com>.
- Swagger UI: <https://sleepsafe-agent-7474648945814205.aws.databricksapps.com/docs>.
- Deployment `01f1b9f12ce41d0b8bb5d2b05d133352` succeeded on 2026-09-26.
  Live `/health`, `/classify/{session_id}`, and `/ask` returned HTTP 200.
- The deployed app uses the bundled sample report. Real CRNN/Delta data, an LLM
  serving endpoint, and a frontend are **not connected**.
- The App compute is running and may incur Databricks charges until stopped.
- The pre-ML preprocessing package is implemented, tested locally, and synced
  to `/Workspace/Users/yahuja2@wisc.edu/sleepsafe-preprocessing`. This is
  separate from the live App source; the existing agent behavior is unchanged.
- Existing UC Volume: `/Volumes/workspace/default/sleepsafe`; `raw/` and
  `prepared/` directories have been created.
- Delta manifest table `workspace.default.sleepsafe_audio_windows` exists
  with the 12 expected metadata columns.
- The existing Serverless Starter Warehouse was used for table DDL and then
  stopped; its verified state after setup is `STOPPED`.
- Manual serverless preprocessing job **SleepSafe audio preprocessing**,
  job ID `641484757512224`, uses a Python script task, serverless environment
  version 4, and librosa/soundfile/numpy dependencies. It reads
  `raw/<session_id>/recording.wav`, writes prepared windows and `manifest.json`
  under `prepared/<session_id>/`, and appends manifest rows to the table.
- No raw recording has been uploaded and the job has not been run. The CRNN
  model/pipeline is not provided or connected. The agent still uses sample
  report data; connecting real CRNN JSON reports to its report/event tables is
  a separate remaining step.

## Files and data flow

1. `app.py` exposes the FastAPI routes. It selects `DeltaRepository` only when
   `SLEEPSAFE_REPOSITORY=delta`; otherwise the sample repository is active.
2. `sleepsafe/repository.py` loads the bundled JSON or queries two Unity Catalog
   Delta tables. `SampleRepository` returns only the bundled session.
3. `sleepsafe/tools.py` validates the report and provides four read-only tools:
   `get_sleep_report`, `get_events`, `get_event_near_time`, and
   `get_audio_quality`.
4. `sleepsafe/classifier.py` computes the classification. `sleepsafe/schema.py`
   validates the report and requires exactly seven output fields.
5. `sleepsafe/agent.py` answers the nine demo questions deterministically. If
   `SLEEPSAFE_LLM_ENDPOINT` is set, other questions can use a Databricks chat
   endpoint with function tools. That endpoint has not been configured or
   live-tested here.
6. `sleepsafe/system_prompt.txt` is the LLM's policy. `demo.py` exercises the
   sample prompts; `tests/test_sleepsafe.py` covers eight classifier/Q&A cases.
7. `app.yaml` starts Uvicorn with `['uvicorn', 'app:app']`. Databricks provides
   `UVICORN_HOST` and `UVICORN_PORT` at runtime. Do not pass
   `${DATABRICKS_APP_PORT}` as a literal CLI argument.
8. `preprocessing/audio.py` independently inspects WAVs, converts to mono,
   resamples, validates, computes basic quality metrics, and splits waveforms.
   `preprocessing/manifest.py` writes PCM-16 windows and produces manifest
   records, with optional Spark DataFrame/Delta writing. The CLI is
   `scripts/preprocess_audio.py`. This layer runs before a future CRNN and is
   separate from the deployed post-ML API.

Bundled session ID: `a3f9c2e1-lapel-0926`. The sample's model summary reports
estimated AHI 13.4, severity `mild`, 58 apnea events, 31 hypopnea events, and
snoring during 34.2% of estimated sleep. Only eight events are listed. The
listed apnea events disagree with the summary count even though
`events_truncated` is false; SleepSafe states this discrepancy instead of
inventing details. The sample classifies as `mixed_respiratory_pattern`.

## Classification contract

`GET /classify/{session_id}` always returns exactly:
`classification`, `severity`, `primary_findings`, `notable_events`,
`data_quality`, `summary`, and `caveats`. `severity` preserves the CRNN's
`severity_estimate`; SleepSafe never recalculates it. Allowed labels are
`no_significant_respiratory_pattern`, `apnea_hypopnea_pattern`,
`snoring_dominant_pattern`, `mixed_respiratory_pattern`, `poor_audio_quality`,
and `insufficient_evidence`.

Current hackathon rules: less than 70% valid audio gives `poor_audio_quality`;
estimated AHI at least 5 plus apnea/hypopnea count above zero is an
apnea/hypopnea pattern; snoring at least 30% is a snoring pattern; both yield
`mixed_respiratory_pattern`. Missing or invalid evidence yields
`insufficient_evidence`. These are project heuristics, not clinical cutoffs.

## Agent behavior and limits

The policy in `sleepsafe/system_prompt.txt` requires factual claims to come
from the read-only tools. The agent must not diagnose disease, infer obstructive
versus central apnea, invent events or history, or override the model's
severity. Trained detections have stronger evidence than `zero_shot`
detections. Relevant answers mention recording quality and caveats. A request
for unavailable prior-night data must say it is unavailable. Classification
comes from the deterministic classifier, never from an LLM improvisation.

`POST /ask` takes `{"session_id":"...","question":"..."}` and returns an
answer. Direct answers cover event times, longest apnea, snoring, recording
quality, classification reasons, detected events, zero-shot detections, hourly
activity, and prior-night comparison. Time questions are interpreted in UTC
because the report has UTC timestamps and no user timezone. For other
questions, the optional LLM can call the four report tools and classifier;
tool calls are forced to the request's session ID. Without an LLM endpoint,
the app returns a description of the supported questions. The LLM path has
not been exercised against a live endpoint.

## Pre-ML audio preprocessing

The preprocessing package accepts WAV input. It records original sample rate,
channel count, frame count, duration, format, and subtype; averages input
channels to mono; resamples to a requested rate (default 16,000 Hz) using
librosa; and validates nonempty finite audio with a positive duration and
nonzero signal. It outputs a one-dimensional mono waveform. It does not
perform MFCC or Mel extraction, diagnosis, event detection, denoising, or
model-specific transforms.

Quality records include duration in seconds, absolute peak amplitude, RMS, and
`low_signal`. The digital silence rejection guard is `peak <= 1e-8`; the
`low_signal` flag is set when RMS is below `1e-3`. These values are explicit
audio-quality heuristics only, not medical thresholds. Completely silent
windows inside an otherwise usable recording are preserved and flagged.

Default windows are 1,200 seconds. The last partial window is retained. Files
are mono WAV with `PCM_16` subtype named `window_000.wav`, `window_001.wav`,
etc. The manifest contains `session_id`, integer `window_id`, `audio_path`,
start/end offset, duration, sample rate, channel count, status, peak, RMS, and
low-signal flag. Status is `ready` or `low_signal`. Offsets are sequential
waveform offsets in seconds.

Local command:

```bash
.venv/bin/python scripts/preprocess_audio.py \
  --input /path/to/recording.wav \
  --session-id test-session-001 \
  --output-dir /path/to/prepared/test-session-001
```

Databricks commands (upload a raw recording, then start the configured manual
serverless job):

```bash
databricks fs cp /path/to/recording.wav \
  dbfs:/Volumes/workspace/default/sleepsafe/raw/test-session-001/recording.wav \
  --profile sleepsafe
databricks jobs run-now 641484757512224 --profile sleepsafe \
  --json '{"job_parameters":{"session_id":"test-session-001"}}'
```

The job and destination Delta table have been created in Databricks. It appends
rows to `workspace.default.sleepsafe_audio_windows`. The JSON manifest is
written to the prepared output directory. CRNN input is each prepared mono
16 kHz WAV plus its row of metadata; there are no spectral features. Raw input
is stored at
`/Volumes/workspace/default/sleepsafe/raw/<session_id>/recording.wav` and
prepared output at `/Volumes/workspace/default/sleepsafe/prepared/<session_id>`.
The job has not yet been run because no real recording is present. The local
Spark/Delta writer was not tested from this machine.

`GET /health` returns `{"status":"ok"}`. Unknown session IDs return 404;
invalid report/question data can return 422. The API has no custom frontend;
`/docs` is the current interface.

## Local development and verification

From the project folder:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python demo.py
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

The local `.venv` has all packages from `requirements.txt`; it is ignored by
Git. On 2026-09-26 all eight tests passed. The local sample route functions
and the deployed HTTP endpoints returned the expected classification and
12:18 AM event answer. Running MLflow locally created `mlflow.db`, which is
also ignored by Git. The preprocessing CLI was run on a generated 60-second
8 kHz stereo WAV: it produced one 60-second 16 kHz mono PCM-16 window and a
JSON manifest under `/private/tmp/sleepsafe-demo-prepared/test-session-001`.
All 11 preprocessing tests and all 8 existing SleepSafe tests passed after
the implementation. Spark/Delta writing was not tested locally because this
environment does not provide PySpark or a Unity Catalog Volume.

## Deploy and operate

After local changes, run from this folder:

```bash
databricks sync . /Workspace/Users/yahuja2@wisc.edu/sleepsafe-agent --profile sleepsafe
databricks apps deploy sleepsafe-agent --source-code-path /Workspace/Users/yahuja2@wisc.edu/sleepsafe-agent --profile sleepsafe
databricks apps get sleepsafe-agent --profile sleepsafe
```

Check a failed deployment with
`databricks apps logs sleepsafe-agent --tail-lines 100 --profile sleepsafe`.
CLI authentication may need refreshing with
`databricks auth login --host https://dbc-0cc75cba-cde3.cloud.databricks.com --profile sleepsafe`.
Do not upload `.venv`, `mlflow.db`, or local credentials. A `databricks sync
--dry-run` on 2026-09-26 showed only the intended project files.

For real reports, follow the Delta table schema and permission setup in
`README.md`, then set `SLEEPSAFE_REPOSITORY=delta` and `DATABRICKS_HTTP_PATH`
in `app.yaml`. This requires a CRNN report pipeline and a SQL warehouse; no
real report source has been supplied. For open-ended LLM questions, supply a
tool-calling serving endpoint, grant the App service principal `CAN QUERY`,
and set `SLEEPSAFE_LLM_ENDPOINT`. These steps remain optional and pending.

The preprocessor job is manual and separate from the App. To use it, upload a
real WAV with the `databricks fs cp` command above, then trigger job
`641484757512224` with a matching `session_id`. The job only prepares WAVs and
metadata; it does not run a CRNN or populate the agent's session/event report
tables. Keep this job manual until a CRNN task and report-writing contract are
available.

## Change history

- 2026-09-26: Created the local virtual environment; installed dependencies;
  passed eight tests and sample API checks.
- 2026-09-26: Added `mlflow.db` and `mlruns/` to `.gitignore` after local
  tracing created a database.
- 2026-09-26: Created and deployed `sleepsafe-agent` in Databricks. The first
  deployment failed because `${DATABRICKS_APP_PORT}` was passed literally;
  changed `app.yaml` to let Uvicorn use Databricks' host/port environment
  variables. The next deployment and live endpoint checks succeeded.
- 2026-09-26: Added this context file and a project instruction to update it
  whenever SleepSafe changes.
- 2026-09-26: Added an independent WAV preprocessing package and CLI for
  inspection, mono/16 kHz standardization, validation, quality metrics,
  20-minute PCM-16 windows, JSON metadata, and optional Spark/Delta manifest
  writing. Added `librosa`, `soundfile`, and `numpy` dependencies and documented
  the use. Verified 11 preprocessing tests, all 8 existing tests, and a local
  60-second stereo CLI run; Spark/Delta and Volume access remain pending.
- 2026-09-26: Set up Databricks preprocessing resources: verified the existing
  Volume, created `raw/` and `prepared/`, created the Delta manifest table,
  synced only the preprocessing package/CLI to a separate workspace folder,
  and created manual serverless job `641484757512224`. Verified the job
  configuration, table schema, upload directory, prepared directory, and
  stopped the SQL warehouse after DDL.
  The job remains unrun pending an actual recording; no CRNN or real report
  integration was supplied.
- 2026-09-26: Connected the local SleepSafe folder to
  `HariharaNalamotu/SleepSafe` on branch `yagya`. Added the source, tests,
  configuration, and documentation to Git; excluded the local Databricks sync
  snapshot and generated environment/data files. Verified the project tests
  before pushing the branch.
