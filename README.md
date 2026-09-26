# SleepSafe

Audio-based sleep screening from a lapel mic. A Raspberry Pi streams 10 s chunks to Databricks;
when the session ends, a job runs the model, writes incidents to Delta tables and has an LLM
write a report for the user.

```
RPi (lapel mic) ──10 s FLAC chunks──► UC Volume chunks/<session>/
             └──on stop: jobs/run-now──► Databricks job (serverless CPU)
                                          ├─ Stage 1: frozen PANNs CNN14 (16 kHz) → per-second features
                                          ├─ Stage 2: BiGRU head → apnea / hypopnea / snore / asleep per second
                                          ├─ post-processing → incidents + summary (night JSON)
                                          ├─ Delta: sessions, incidents, reports
                                          └─ LLM (Foundation Model API) → Markdown report
```

## Model

| Part | What | Trained? |
|---|---|---|
| Encoder | PANNs CNN14 16 kHz (AudioSet), GPU-native log-mel front end | frozen |
| Features / s | 2048-d embedding (PCA → 256), 12 AudioSet scores, 24 band-energy stats | – |
| Head | Linear → Conv1d → 2-layer BiGRU (128) → 4 sigmoid outputs, ~0.6 M params | yes |
| Zero-shot flags | cough, gasp, snort, wheeze, speech from AudioSet scores (data-calibrated thresholds) | no |

Training data: PSG-Audio (Korompili et al., 2021, CC BY 4.0) — hospital polysomnography with
synchronized ambient + tracheal mics and technician-scored apneas/hypopneas, snore events and
sleep stages. No manual annotation. Lapel-mic conditions are simulated with GPU augmentation
(muffling, clothing rustle, room noise, gain, dropouts). Split by subject, stratified by AHI.

## Using the trained model (no retraining needed)

The trained head is committed at `runs/v2/model.pt` (test results in `runs/v2/metrics.json`).
Fetch the frozen encoder weights (358 MB, too large for GitHub) once:

```bash
pip install torch numpy scipy soundfile requests
python scripts/fetch_encoder.py                     # -> data/pretrained/Cnn14_16k.pth
python scripts/run_session.py recording.flac --out night.json --device cpu
```

## Local workflow (Windows, RTX GPU)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe numpy scipy requests huggingface_hub soundfile
python scripts/download_convert.py --n-subjects 80        # ~275 GB streamed, kept as 16 kHz npy
.venv\Scripts\python scripts/extract_features.py           # GPU, ~1 min per subject (6 variants)
.venv\Scripts\python scripts/train.py --name v1            # GPU-resident training, minutes
.venv\Scripts\python scripts/run_session.py rec.flac --out night.json [--demo-mode]
```

## Databricks deployment (Free Edition)

Live workspace: `https://dbc-0cc75cba-cde3.cloud.databricks.com` (catalog `workspace`, schema `default`).

| Piece | Where |
|---|---|
| Audio + models | volume `/Volumes/workspace/default/sleepsafe` (`chunks/<session>/`, `raw/<session>/recording.wav`, `models/`, `reports/`) |
| Pipeline code | `/Workspace/Users/yahuja2@wisc.edu/sleepsafe-model` |
| Processing job | `sleepsafe-process-session` (serverless CPU; report LLM `databricks-gpt-oss-120b`) |
| Tables | `sleepsafe_sessions` (report_json), `sleepsafe_events` (event_json), `sleepsafe_reports` (Markdown) |
| Dashboard views | `sleepsafe_v_sessions`, `sleepsafe_v_events`, `sleepsafe_v_event_counts` |
| API (Databricks App) | `https://sleepsafe-agent-7474648945814205.aws.databricksapps.com` (`/docs`) |

Redeploy after changing pipeline code or the model (idempotent):

```bash
export DATABRICKS_HOST=https://dbc-0cc75cba-cde3.cloud.databricks.com DATABRICKS_TOKEN=<token>
python scripts/fetch_encoder.py
python scripts/deploy_databricks.py        # tables, views, model upload, code, job
```

Process a session:

```bash
# live, from the Pi (watches a folder of 10 s chunks; triggers the job when stopped)
python pi/uploader.py --watch-dir /path/to/chunks --volume /Volumes/workspace/default/sleepsafe     --job-id <job id> --duration-min 30 [--demo-mode]
# or replay a finished recording through the same path (for the time-lapse demo)
python scripts/simulate_pi.py recording.wav --session-id demo-night-1 --job-id <job id> --wait
```

Capture must be 16 kHz mono (any rate is resampled) **without** noise suppression, noise gates or AGC.

### Dashboard data

**REST (Databricks App, read-only):**

| Route | Returns |
|---|---|
| `GET /sessions` | one summary row per session, newest first |
| `GET /sessions/{id}/dashboard` | recording, summary, classification, event counts, timeline, written report, caveats |
| `GET /sessions/{id}` | full report JSON |
| `GET /sessions/{id}/events?type=apnea` | events |
| `GET /sessions/{id}/report` | LLM-written Markdown report |
| `GET /classify/{id}`, `POST /ask` | the agent's classification and Q&A |

Databricks Apps require a signed-in Databricks user (or an OAuth token), so a dashboard should run
as a Databricks App itself or call the API with an OAuth bearer token.

**SQL (works with a personal access token):** `POST /api/2.0/sql/statements` on warehouse
`66987446ea53f550`, e.g.

```sql
SELECT * FROM workspace.default.sleepsafe_v_sessions ORDER BY start_utc DESC;
SELECT * FROM workspace.default.sleepsafe_v_events WHERE session_id = :sid ORDER BY start_offset_s;
SELECT * FROM workspace.default.sleepsafe_v_event_counts WHERE session_id = :sid;
SELECT report_markdown FROM workspace.default.sleepsafe_reports WHERE session_id = :sid;
```

## Limitations

Screening aid, not a diagnosis. Trained on hospital mics (not lapel), hypopneas are hard without
SpO2, snore labels are device-scored (sparse), zero-shot flags are uncalibrated against labels,
and 30-minute sessions give only a rough events-per-hour estimate.
