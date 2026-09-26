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

## Databricks (Free Edition) setup

1. **Catalog objects** (SQL editor):
   ```sql
   CREATE SCHEMA IF NOT EXISTS workspace.sleepsafe;
   CREATE VOLUME IF NOT EXISTS workspace.sleepsafe.data;
   ```
2. **Upload models** to `/Volumes/workspace/sleepsafe/data/models/`: `data/pretrained/Cnn14_16k.pth`
   and `runs/v2/model.pt` (Catalog Explorer → volume → Upload).
3. **Code**: import this repo into the workspace (Workspace → Create → Git folder, or upload
   `sleepsafe/` and `databricks/` to `/Workspace/Users/<you>/SleepSafe/`).
4. **LLM endpoint**: open *Serving*, pick an available pay-per-token chat model (Claude if listed),
   and put its name in `databricks/job.json` (`--llm_endpoint`).
5. **Job**: fill in `<YOUR_EMAIL>` and the endpoint in `databricks/job.json`, then
   `databricks jobs create --json @databricks/job.json` (or create it in the UI with the same settings).
   If the `download.pytorch.org` index is blocked, replace the two torch lines with plain `torch`.
6. **Token**: Settings → Developer → Access tokens; used by the Pi.
7. **Pi**: `pip install requests`, then
   ```bash
   export DATABRICKS_HOST=https://<workspace-host> DATABRICKS_TOKEN=<token>
   python pi/uploader.py --watch-dir /path/to/chunks --volume /Volumes/workspace/sleepsafe/data \
       --job-id <job id> --duration-min 30
   ```
   Capture must be 16 kHz mono (or any rate — it's resampled), **without** noise suppression,
   noise gates or AGC.

### Dashboard queries

```sql
-- latest session summary
SELECT session_id, start_utc, duration_s/60 AS minutes, events_per_hour, severity_estimate,
       snore_pct_of_sleep, counts_json
FROM workspace.sleepsafe.sessions ORDER BY processed_utc DESC LIMIT 1;

-- incident timeline for a session
SELECT start_utc, type, duration_s, confidence, confidence_basis
FROM workspace.sleepsafe.incidents WHERE session_id = :session_id ORDER BY start_offset_s;

-- report
SELECT report_markdown FROM workspace.sleepsafe.reports WHERE session_id = :session_id;
```

## Limitations

Screening aid, not a diagnosis. Trained on hospital mics (not lapel), hypopneas are hard without
SpO2, snore labels are device-scored (sparse), zero-shot flags are uncalibrated against labels,
and 30-minute sessions give only a rough events-per-hour estimate.
