# SleepSafe — full project context

Last updated: 2026-09-27 (UTC). Covers everything from the first design discussion to the verified
end-to-end system. **This file contains no secret values.** Every credential is listed by name, purpose
and storage location only (section 13). Keep it that way: never paste a token, password or key here.

---

## 0. One-paragraph summary

SleepSafe is a hackathon project that screens sleep for breathing problems from audio. A person wears a
Bluetooth headset microphone connected to a Raspberry Pi. From a web dashboard (Vercel) you press
**Start recording**; the Pi streams 10-second audio chunks to a Databricks volume while it records. When
you press **Stop**, the Pi finishes uploading and starts a Databricks job that runs a trained audio model
(frozen PANNs CNN14 encoder + a small BiGRU head trained on the PSG-Audio hospital dataset). The job
writes per-event incidents and a session summary to Delta tables and has an LLM
(`databricks-gpt-oss-120b`) write a plain-language, non-diagnostic report. The dashboard shows the
sessions, an interactive event timeline, the events table and the written report. A teammate's
Databricks App (`sleepsafe-agent`) exposes the same data plus classification and Q&A. It is a
**screening aid, not a diagnosis**.

---

## 1. People, accounts, ownership

| Who | Role | Accounts used |
|---|---|---|
| Harihara Nalamotu (`harihara.nalamotu@gmail.com`, GitHub `HariharaNalamotu`) | Model, pipeline, Databricks job, dashboard, Pi agent (this repo, branch `hari`) | GitHub repo owner; Vercel account `hariharanalamotu` (team scope `hariharanalamotus-projects`) |
| yahuja2 (`yahuja2@wisc.edu`) | Built the `sleepsafe-agent` Databricks App (classification + Q&A); owns the Databricks workspace | Databricks Free Edition workspace owner |
| Nitesh | Built the Pi audio capture (`audio_stream.py`, branch `nitesh`) and the Pi runbook | Code lives on the Pi at `~/SleepSafe` and in `~/sleepsafe-nitesh.bundle` |

- GitHub repo: `https://github.com/HariharaNalamotu/SleepSafe` (private). Branches: `main` (Vercel production
  branch), `hari` (all of this work), `nitesh` (Pi capture; present on the Pi as a checkout + bundle).
- Databricks workspace: `https://dbc-0cc75cba-cde3.cloud.databricks.com`, workspace id `7474648945814205`,
  **Free Edition** (serverless only, no GPUs, 5 concurrent job tasks, fair-usage quota, non-commercial use,
  no compliance enforcement → not for real patient data).
- Vercel project: `sleepsafe-dashboard` (id `prj_iOCgtk4Tvjo7Pgyype9IbdDY0VxU`), production URL
  `https://sleepsafe-dashboard.vercel.app` (alias `sleepsafe-dashboard-hariharanalamotus-projects.vercel.app`).

---

## 2. Timeline (what happened, in order)

1. **Design discussion.** Sound-event-detection framing (CNN + RNN over log-mels), per-second multi-label
   outputs, post-processing to timestamped events, JSON for an LLM report agent. Constraints that shaped
   everything: no time to annotate data; ~20 h budget; Databricks Free Edition; mic is a wearable
   (first "lapel", later confirmed **external Bluetooth headset**); a 30-minute recorded session, time-lapsed
   for the demo, with the report shown on a dashboard; latency does not matter (report at end of session).
2. **Architecture chosen.** Frozen pretrained encoder (PANNs CNN14 16 kHz, AudioSet) → cached per-second
   features → small trained BiGRU head. Labels come free from PSG-Audio's technician-scored annotations.
   10-minute windows, bidirectional; whole session processed once at the end.
3. **Data.** PSG-Audio V3 from a Hugging Face mirror; 80 subjects selected, 1 dropped (corrupt at source),
   79 used (~85 GB of 16 kHz audio after conversion).
4. **Feature extraction** on the laptop GPU (RTX 5080 Laptop, 16 GB): 6 variants per subject
   (ambient mic, tracheal mic, each clean + 2 lapel-style augmentations).
5. **Laptop crashes.** Repeated blue screens (see section 12): `0x133 DPC_WATCHDOG_VIOLATION` in
   `nvlddmkm.sys` (NVIDIA driver interrupt handler), plus two `0x20001 HYPERVISOR_ERROR`. Driver reinstall
   (596.49 → 617.14), BIOS update (Q7CN40WW → Q7CN78WW), Memory Integrity off, Proton VPN uninstalled:
   crashes persisted, including at idle in dGPU-only mode. Work moved to CPU where possible.
6. **Models.** v1 (22 training subjects, GPU) then **v2** (59 training subjects, trained on CPU to step
   1000, finalized on CPU). v2 is the production model.
7. **Databricks deployment** into the teammate's existing workspace, reusing his `default` schema,
   `sleepsafe` volume and his app's expected tables; processing job created; teammate app switched from its
   bundled sample to real Delta data and given dashboard routes.
8. **Vercel dashboard** built (Next.js), password-protected, deployed.
9. **Raspberry Pi** reached over USB-C (Windows needed the RNDIS driver forced), then moved to **Wi-Fi**
   (Ubuntu 24.04 + Broadcom chip needed plain WPA2-PSK). Nitesh's recorder reused unchanged; a new
   **Pi agent** added so the dashboard can start/stop sessions through Databricks.
10. **Domain-shift fix.** Real headset audio (speech) was misread as snoring/sleep; added AudioSet-based
    sanity gates (speech ⇒ awake; snoring needs AudioSet evidence), validated on held-out PSG data.
11. **Full end-to-end test passed**: dashboard Start → Pi sample playback through the real capture/DSP
    chain → live 10 s uploads → auto-stop → Databricks job → dashboard shows results (section 11).

---

## 3. System architecture

```
                          (control plane: files in the Databricks volume)
 Vercel dashboard ── PUT control/pi-01/command.json ──►  Databricks UC volume  ◄── poll every 3 s ── Pi agent
 (Next.js, server-side      GET control/pi-01/status.json ◄──────────────────────── publish ─────── (sleepsafe-pi-agent)
  Databricks calls only)                                                                               │ starts/stops
                                                                                                       ▼
 Bluetooth headset (HFP mSBC 16 kHz) or sample WAV → PipeWire → FFmpeg (16 kHz mono, HP 60 Hz, LP 7 kHz)
   → audio_stream.py (Nitesh): exact 10 s WAV chunks → local spool → Files API upload (live, ~1 s after capture)
   → /Volumes/workspace/default/sleepsafe/chunks/<session_id>/<YYYYMMDD>/<UTC-completion-time>_pi-01_<uuid>.wav

 On Stop: agent stops recorder → uploads leftovers → POST jobs/run-now (sleepsafe-process-session)
   Databricks job (serverless CPU): assemble chunks by timestamp → CNN14 encoder → preprocessor → BiGRU head
   → AudioSet gates → post-processing → night JSON → Delta (sleepsafe_sessions / _events) + reports/<sid>.json
   → LLM (databricks-gpt-oss-120b) → Markdown report → sleepsafe_reports + reports/<sid>.md

 Readers: Vercel dashboard (SQL Statement API)  ·  teammate's Databricks App sleepsafe-agent (SQL connector)
```

Design principles that matter:
- The Pi makes **outbound HTTPS only** → works on any Wi-Fi, no port forwarding, no inbound SSH needed.
- The dashboard's Databricks token lives **server-side** in Vercel; the browser never sees it.
- The recorder (Nitesh's code) is **unmodified**; the agent switches it into cloud mode per session with a
  systemd drop-in and removes the drop-in afterwards.
- Heavy compute (inference) runs on **Databricks serverless CPU**, not the laptop GPU.

---

## 4. Repository layout (`D:\SleepSafe`, branch `hari`)

| Path | Purpose |
|---|---|
| `README.md` | User-facing overview, local workflow, Databricks deployment, dashboard data access |
| `context.md` | This file |
| `.gitignore` | Ignores `data/`, `scratch/`, `.venv/`, `runs/` (except force-added v2 files), `*.log`, `.playwright-mcp/`, `.vercel/` |
| `.vercelignore` | Uploads only `dashboard/` to Vercel (prevents uploading the 85 GB `data/`) |
| `sleepsafe/labels.py` | Parses PSG-Audio RML → per-second targets (apnea, hypopnea, snore, asleep) + loss masks |
| `sleepsafe/encoder.py` | GPU/CPU-native PANNs CNN14 16 kHz: log-mel from the checkpoint's own mel filterbank, 64 s segments with 3.2 s context, per-second 2048-d embedding + 12 AudioSet scores + 24 band-energy stats; `normalize_loudness` (session p90 RMS → −20 dBFS) |
| `sleepsafe/augment.py` | GPU "lapel mic" augmentation: muffling low-pass, colored room noise, clothing rustle bursts, gain, dropouts |
| `sleepsafe/model.py` | `Preprocessor` (PCA 2048→256, logit scores, energy minus 30-min block median, standardization) + `SleepHead` (BiGRU) + `predict_session` (10-min windows, 1-min margins) |
| `sleepsafe/postprocess.py` | `apply_audio_gates`, `detect_events`, `signal_quality`, `session_summary` (the night JSON) |
| `sleepsafe/pipeline.py` | `SleepSafePipeline` (end-to-end inference), `load_audio`, `assemble_chunks` (index-named), `assemble_timed_chunks` + `chunk_start_utc` (Pi timestamp-named chunks) |
| `scripts/download_convert.py` | Downloads PSG-Audio V3 EDFs from HF, keeps Mic + Tracheal at 16 kHz int16 `.npy`, deletes EDFs |
| `scripts/extract_features.py` | Stage-1 features for all subjects (6 variants), low-VRAM mode |
| `scripts/train.py` | Trains/evaluates the head; `--device cpu|cuda`, `--sleep-ms` duty cycle, `--patience` early stop, `--finalize-from` (tune + test a checkpoint without training); saves `model_partial.pt` on each improvement |
| `scripts/run_session.py` | Local end-to-end run on a file or a chunk folder → night JSON |
| `scripts/fetch_encoder.py` | Downloads `Cnn14_16k.pth` (358 MB; too big for GitHub) from the HF mirror |
| `scripts/deploy_databricks.py` | Idempotent Databricks deploy: tables, views, model upload, code import, LLM selection, job create/reset |
| `scripts/simulate_pi.py` | Replays a recording through the Pi upload path (index-named chunks) and triggers the job |
| `databricks/process_session.py` | The Databricks job entry point (see section 7) |
| `dashboard/` | Next.js dashboard (section 8) |
| `pi/agent.py`, `pi/sleepsafe-pi-agent.service`, `pi/uploader.py`, `pi/README.md` | Pi agent + systemd unit; legacy stand-alone uploader; Pi docs |
| `runs/v2/model.pt`, `runs/v2/metrics.json` | Production model (force-added to git, 4.6 MB) and its metrics/split |
| `data/` (not in git) | `audio16k/` (85 GB), `features/` (~35 GB), `rml/`, `pretrained/Cnn14_16k.pth`, `class_labels_indices.csv`, `subjects.json` |
| `runs/` (not in git except v2) | `v1/`, `v2/` (incl. `model_partial.pt`), `interim/`, `smoke/`, `v2gpu/` (empty; crashed) |
| `scratch/` (not in git) | Secrets files (section 13), test clips (`demo30.flac`, `demo_chunks/`, `psg_apnea_8min.wav`, `pi_real/`), screenshots, logs, teammate code copies (`teammate/`, `teammate_orig/`), helper scripts (`rndis_fix.ps1`, `hotspot.ps1`, `analyze_dumps.ps1`, `gpu_throttle.ps1`) |

Commits on `hari` (newest first at time of writing): `b71c81d` Pi control + gates, `c093651` Vercel root dir,
`8388e66` dashboard, `b8ac28a` Databricks integration, `ac2fa40` (setup notebook, later removed),
`37f81f2` v2 model + fetch script, `7b71a80` initial pipeline. (Check `git log` for later ones.)

---

## 5. Data

- **Dataset:** PSG-Audio (Korompili et al., Scientific Data 2021, CC BY 4.0). Hospital polysomnography with
  synchronized **ambient microphone** (`Mic`, EDF channel 18, 48 kHz) and **tracheal microphone**
  (`Tracheal`, channel 19, 48 kHz); technician-scored respiratory events, device-scored snore events,
  sleep stages in RML (XML) files.
- **Source used:** Hugging Face partial mirror `dust-systems/psg-audio` (~586 GB; original ~986 GB on
  scidb.cn behind expiring FTP credentials). V3 EDFs are split into 1-hour parts; file names literally
  contain `%5B001%5D`, so URLs must escape `%` as `%25`.
- **Selection:** 80 random complete subjects (seed 0) → `data/subjects.json`; `00001339-100507` removed
  (part 2 corrupt at source: zero header, truncated). **79 subjects.**
- **Conversion:** Mic + Tracheal → 16 kHz int16 `.npy` per 1-hour part (`resample_poly` 1:3).
- **Labels (`sleepsafe/labels.py`):** apnea = Obstructive/Mixed/Central; hypopnea; snore (Nasal/Snore,
  device-scored and sparse → unreliable ground truth); asleep = N1/N2/N3/REM. Masks: respiratory loss only
  during scored sleep and not within ±2 s of event boundaries; snore not within ±2 s of snore boundaries.
- **Split (subject-level, stratified by AHI):** sorted by AHI, `i % 8 == 0` → test, `i % 8 == 4` → val,
  rest train → 59 / 10 / 10. Exact lists in `runs/v2/metrics.json`.

---

## 6. Model

**Stage 1 (frozen):** PANNs `Cnn14_16k_mAP=0.438` (checkpoint from HF mirror `niobures/PANNs`,
`models/pretrained/`). Implemented natively (`sleepsafe/encoder.py`), bf16 on GPU. Per second: 2048-d
embedding (fc1+ReLU on frames ×2, averaged), 12 AudioSet probabilities (speech 0, breathing 41, wheeze 42,
snoring 43, gasp 44, pant 45, snort 46, cough 47, throat clearing 48, sniff 50, rustle 487, silence 500;
max per second), 24 band-energy stats (8 mel bands × mean/max/std). Loudness normalized per session
(p90 per-second RMS → −20 dBFS, max +40 dB).

**Stage 2 (trained, ~0.62 M params):** input = PCA-256 embedding + 12 score logits + 24 energy features
(mean/max relative to the 30-min block median), standardized → Linear→LayerNorm→GELU → Conv1d k5 residual
→ 2-layer BiGRU (128/direction) → 4 sigmoid outputs per second: apnea, hypopnea, snore, asleep. Trained on
10-min crops, AdamW lr 2e-3, OneCycle, masked BCE with sqrt pos_weight, loss weights 1/1/0.5/0.5.

**Post-processing:** 5 s median filter; thresholds tuned on validation (v2: apnea 0.2, hypopnea 0.5,
snore 0.5, asleep 0.5); merge gaps ≤3 s; apnea/hypopnea ≥10 s; snore runs merged into episodes (≤30 s gap);
wake periods ≥60 s; breathing events dropped during predicted wake unless `demo_mode`; zero-shot flags
(cough, gasp, snort, wheeze, speech) with data-calibrated thresholds (99.9th percentile).

**AudioSet sanity gates (`apply_audio_gates`), added after the headset test:** smoothed (15 s) AudioSet
speech ≥0.5 ⇒ awake, no snore, no apnea/hypopnea; snore requires 5 s-median AudioSet snoring ≥0.05.
On held-out PSG test: snore F1 0.42 → 0.47; touched <0.4 % of asleep/apnea seconds.

**Results (held-out test, 10 subjects, v2):**

| Setting | Apnea+hypopnea event F1 | Precision / recall | AHI Pearson | AHI MAE | Severity exact / ±1 |
|---|---|---|---|---|---|
| Tracheal | 0.835 | 0.86 / 0.81 | 0.91 | 6.6 | 80 % / 90 % |
| Tracheal + lapel aug | 0.826 | 0.85 / 0.81 | 0.91 | 7.0 | 70 % / 90 % |
| Ambient mic | 0.801 | 0.84 / 0.76 | 0.90 | 7.8 | 80 % / 90 % |
| Ambient + lapel aug | 0.787 | 0.83 / 0.74 | 0.93 | 7.0 | 80 % / 90 % |

30-minute sessions score the same as full nights. Sleep/wake accuracy 91–93 %. Hypopnea is weak
(AP 0.18–0.30; audio has no SpO2). v1 (22 train subjects) scored 0.48–0.58 F1.

**Caveats:** trained on hospital mics, never validated on the headset; headsets may apply their own noise
suppression in HFP mode; hypopnea limited; snore labels device-scored; zero-shot thresholds uncalibrated
against labels; short sessions extrapolate events/hour.

**Night JSON (`session_summary`)** keys: `session_id`, `model{name,version,encoder,demo_mode}`,
`recording{start_utc,end_utc,duration_s,valid_audio_s,excluded_s,estimated_sleep_s,sleep_onset_offset_s}`,
`summary{estimated_ahi,respiratory_events_per_hour,rate_basis,severity_estimate,severity_bands,short_session,
counts,snore_pct_of_sleep,longest_apnea_s,events_per_hour_by_hour}`, `events[{id,type,start_offset_s,
end_offset_s,duration_s,start_utc,end_utc,confidence,confidence_basis,evidence?,reason?}]`,
`events_truncated`, `caveats[]`. Matches the teammate app's schema.

---

## 7. Databricks

| Resource | Value |
|---|---|
| Catalog / schema | `workspace.default` |
| Volume | `/Volumes/workspace/default/sleepsafe` — `models/` (`model.pt`, `Cnn14_16k.pth`), `chunks/<session>/`, `raw/<session>/recording.wav` (teammate's convention), `reports/<session>.json|.md`, `control/pi-01/` (`command.json`, `status.json`), plus teammate's `prepared/`, `raw/`, `sleepsafe_test.wav` |
| Tables (read by apps) | `sleepsafe_sessions(session_id, report_json)`, `sleepsafe_events(session_id, start_offset_s, event_json)`, `sleepsafe_reports(session_id, report_markdown, llm_endpoint, created_utc)` |
| Dashboard views | `sleepsafe_v_sessions`, `sleepsafe_v_events`, `sleepsafe_v_event_counts` |
| SQL warehouse | `Serverless Starter Warehouse`, id `66987446ea53f550`, 2X-Small, auto-stops (first query after idle ~10–20 s) |
| Processing job | `sleepsafe-process-session`, id **769690655470224**, serverless CPU, env `environment_version 4`, deps `torch==2.11.0+cpu` (extra index download.pytorch.org/whl/cpu), numpy, scipy, soundfile, openai, databricks-sdk; params `session_id`, `demo_mode` |
| Job code | `/Workspace/Users/yahuja2@wisc.edu/sleepsafe-model/` (`sleepsafe/*.py`, `databricks/process_session.py`) |
| Report LLM | `databricks-gpt-oss-120b` (no Claude on this workspace). Prompt forbids naming disorders/syndromes or labeling obstructive vs central |
| Teammate app | `sleepsafe-agent`, `https://sleepsafe-agent-7474648945814205.aws.databricksapps.com` (`/docs`), service principal client id `43c9475b-7f62-4cc1-882e-b1bc43465d6c` (granted USE CATALOG/SCHEMA, SELECT on the tables/views, CAN_USE on the warehouse) |
| Teammate's other assets (untouched) | `sleepsafe_audio_features`, `sleepsafe_audio_windows` tables; job `SleepSafe audio preprocessing` (id 641484757512224); folders `SleepSafe/`, `sleepsafe-preprocessing/` |

**Job behavior (`databricks/process_session.py`):** reads `chunks/<sid>/` recursively; Pi timestamp-named
chunks are placed by time (joins within 1 s seamless, larger gaps = invalid silence); else index-named
chunks; else `raw/<sid>/recording.wav`. Runs the pipeline on CPU (~107 s for 30 min), deletes previous rows
for the session, writes the tables and `reports/<sid>.json`, calls the LLM, writes the report. Typical total
2.5–3 min.

**Teammate app changes made (in the workspace, not in his Mac repo):** `app.yaml` →
`SLEEPSAFE_REPOSITORY=delta`, `DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/66987446ea53f550`; dashboard routes
`/sessions`, `/sessions/{id}`, `/sessions/{id}/events`, `/sessions/{id}/report`, `/sessions/{id}/dashboard`;
CORS; repository list/report methods with sample fallback; his `CONTEXT.md` updated. Originals backed up in
`scratch/teammate_orig/`. **He must pull these into his local project before his next `databricks sync`.**
The app only accepts signed-in Databricks users (PATs get 302).

**Redeploy pipeline code/model:**
```
set DATABRICKS_HOST=https://dbc-0cc75cba-cde3.cloud.databricks.com
set DATABRICKS_TOKEN=<setup token — scratch/.dbtoken>
set DATABRICKS_USER=yahuja2@wisc.edu
.venv\Scripts\python scripts\deploy_databricks.py
```

---

## 8. Dashboard (Vercel)

- URL: `https://sleepsafe-dashboard.vercel.app`, HTTP Basic auth, username `sleepsafe`, password in
  `scratch/.dashboard_password` and Vercel env `DASHBOARD_PASSWORD` (middleware returns 503 if unset).
- Stack: Next.js 15 (App Router), React 19, `react-markdown` + `remark-gfm`, TypeScript; `server-only`
  guards the Databricks clients.
- Pages: `/` (Recorder panel + sessions table), `/sessions/[id]` (stat tiles, event timeline with lanes and
  hover/keyboard tooltips, hourly bars when >1 h, events table, written report, limitations).
- API: `GET /api/sessions`, `GET /api/sessions/{id}`, `GET /api/device`, `POST /api/device`
  `{action:"start"|"stop", source:"headset"|"sample", demo_mode}`.
- Data access: SQL Statement Execution API (views/tables) + Files API (control files), token from env.
- Env vars (Vercel, production): `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (v2 token: sql + files),
  `DATABRICKS_WAREHOUSE_ID`, `DASHBOARD_USER`, `DASHBOARD_PASSWORD`; optional `SLEEPSAFE_VOLUME`,
  `SLEEPSAFE_DEVICE_ID`, `SLEEPSAFE_TABLE_PREFIX`.
- Design: validated categorical palette (apnea blue, hypopnea orange, snoring aqua, other sounds yellow,
  context gray), severity as status icon + label, light/dark, no horizontal scroll at 400 px.
- Deploy: from repo root `npx vercel deploy --prod` (project root directory = `dashboard/`). The project
  is Git-connected: pushes to `main` → production, pushes to `hari` → previews (previews have no env vars,
  so they fail closed with 503).

---

## 9. Raspberry Pi

| Item | Value |
|---|---|
| Hardware / OS | Raspberry Pi 4 Model B Rev 1.5, 4 GB; Ubuntu 24.04.5 LTS aarch64, kernel 6.8.0-1064-raspi; Python 3.12; hostname `pi`; user `pi` |
| Wi-Fi | `Theory_Madison_Residence` (WPA2-Personal), IP **172.16.64.177** (DHCP; may change). Fallback network `SleepSafe-Pi` (laptop Windows Mobile Hotspot, WPA2, 2.4 GHz; off by default) |
| USB-C | Ethernet gadget (RNDIS, `usb0` 10.55.0.1/24, runs DHCP; laptop gets 10.55.0.5) |
| Wi-Fi MAC | `2c:cf:67:49:78:f9` |
| SSH | Key-only (`PasswordAuthentication no` via `/etc/ssh/sshd_config.d/50-cloud-init.conf`). Laptop key `C:\Users\harih\.ssh\id_ed25519` is authorized. `ssh pi@172.16.64.177` |
| Headset | Bowers & Wilkins **Px7 S2e**, MAC `EC:66:D1:C6:42:6A`, profile `headset-head-unit-msbc`, source `bluez_input.EC_66_D1_C6_42_6A.0`. **Currently not paired** — re-pair with `bluetoothctl` (pairing mode on the headset) before a real session |
| Recorder (Nitesh) | `~/SleepSafe/audio_stream.py`, user service `sleepsafe-audio` (disabled by default; `local-only.conf` drop-in = local mode), `~/SleepSafe/.env` (source, device id, spool, filters — no secrets), helpers `~/.local/bin/sleepsafe-reconnect`, `sleepsafe-start-local`, runbook `~/sleepsafe-deployment/RUNBOOK.md` |
| Agent (this repo) | `~/sleepsafe-agent/agent.py`, user service `sleepsafe-pi-agent` (enabled, linger on), config `~/.config/sleepsafe/cloud.env` (600), state `~/.local/state/sleepsafe/`, samples `~/sleepsafe-agent/samples/psg_apnea_8min.wav`, per-session drop-in `zz-sleepsafe-session.conf` (exists only while recording), per-session spool `~/SleepSafe/spool-sessions/<sid>/` |
| Wi-Fi config | `/etc/netplan/60-sleepsafe-wifi.yaml` (root 600, both networks, `auth.key-management: psk`), `/etc/modprobe.d/sleepsafe-brcmfmac.conf` (`options brcmfmac feature_disable=0x82000`) |

**Control protocol.** Command (`control/pi-01/command.json`):
`{id, action: start|stop, session_id, source: headset|sample, demo_mode, requested_at}` — the agent acts
once per new `id` (persisted, so restarts don't replay). Status (`control/pi-01/status.json`), every 10 s and
on changes: `{device_id, state: idle|starting|recording|stopping|uploading|processing|error, session_id,
source, demo_mode, started_at, chunks_uploaded, chunks_pending, capture_service, job_run_id, message,
error, last_session{session_id,result,finished_at}, network{ssid,ip}, updated_at}`. Dashboard treats a
status older than 30 s as offline. Sample sessions auto-stop when the sample ends.

**Operate:**
```
systemctl --user status sleepsafe-pi-agent      # agent
journalctl --user -u sleepsafe-pi-agent -f
journalctl --user -u sleepsafe-audio -f         # "Captured …" / "Uploaded …" every 10 s while recording
```

---

## 10. Laptop environment

- Lenovo Legion (model 83F5), Windows 11 Home 26200, RTX 5080 Laptop 16 GB (driver 617.14 clean-installed),
  31 GB RAM, D: drive ~300 GB free at start.
- Python: project venv `D:\SleepSafe\.venv` (uv, CPython 3.12; torch 2.11.0+cu128, torchaudio, numpy,
  scipy, soundfile, requests, huggingface_hub, paramiko, fastapi/pydantic/httpx for testing the teammate app).
  Anaconda base Python has a broken torch (OpenMP clash) — used only for the downloader.
- Node 24 / npm 11; Vercel CLI 60 (logged in as `hariharanalamotu`).
- WinDbg installed (`Microsoft.WinDbg`) for crash-dump analysis.
- Pi USB networking required forcing Windows' inbox `rndiscmp.inf` driver onto `USB\VID_0525&PID_A4A2`
  (`scratch/rndis_fix.ps1`, elevated).

---

## 11. Verified end-to-end test (2026-09-27)

1. Dashboard → Source "Sample recording (PSG apnea clip, 8 min)" → **Start recording** (01:00:36 UTC).
2. Pi agent: null sink `sleepsafe_sample`, `paplay` sample, recorder in cloud mode;
   FFmpeg `aformat=16000 mono, highpass=f=60, lowpass=f=7000`; 48 chunks captured, each uploaded ~1 s later.
3. Sample ended 01:08:49 → auto-stop → drain → job run `1018330013081766` → SUCCESS 01:11:24.
4. Dashboard: session `sample-20260927-010036` listed; 480/480 s valid; 5 apneas + 3 hypopneas;
   **technicians scored 7 events in that audio, model caught 6** (8 flagged); report generated.
Earlier tests: `demo-psg-00001629` (30-min PSG clip via `simulate_pi.py`: 15 scored, 10 flagged, 8 caught);
Nitesh's real headset recording (8 min, speech): no breathing events, speech + cough detected after the gates.

---

## 12. Known issues

- **Laptop instability:** `0x133` in `nvlddmkm.sys` (and `0x20001`), persisted after driver 617.14, BIOS
  Q7CN78WW, Memory Integrity off; happened even at idle in dGPU-only mode and when throttled. Untested:
  Hybrid GPU mode (Lenovo Vantage), turning off hardware-accelerated GPU scheduling, a game stress test.
  If games also crash → Lenovo warranty. Dumps in `C:\Windows\Minidump`. Avoid GPU work on this laptop.
- **Pi Wi-Fi:** netplan's default key management (`WPA-PSK WPA-PSK-SHA256 SAE` + PMF) fails on this chip
  (association status 16); fixed with explicit `psk`. Windows Mobile Hotspot turns itself off when idle.
- **Headset not paired** (needed for real sessions). Headset audio untested for sleep; domain shift likely.
- **Teammate app changes** live only in the workspace until he pulls them.
- **Vercel previews** (branch `hari`) have no env vars → 503 by design.
- **Free Edition:** non-commercial only, no compliance features — prototype data only.

---

## 13. Credentials — names, purposes and locations (values intentionally omitted)

| Credential | Purpose / scopes | Where it is |
|---|---|---|
| Databricks token `sleepsafe-setup-and-pi` | Setup & deploy (files, jobs, sql, workspace, unity-catalog, model-serving, model-serving-inference); 30 days | `D:\SleepSafe\scratch\.dbtoken` |
| Databricks token `sleepsafe-apps-deploy` | Redeploy teammate app, grants (apps, sql, unity-catalog, workspace, access-management, identity); 7 days | `scratch\.dbtoken_apps` |
| Databricks token `sleepsafe-dashboard-vercel` | Old SQL-only dashboard token — **unused, revoke** | `scratch\.dbtoken_dashboard` |
| Databricks token `sleepsafe-dashboard-vercel-v2` | Vercel dashboard (sql, files); 60 days | `scratch\.dbtoken_dashboard2`; Vercel env `DATABRICKS_TOKEN` |
| Databricks token `sleepsafe-pi-device` | Pi agent + recorder (files, jobs); 60 days | `scratch\.dbtoken_pi`; Pi `~/.config/sleepsafe/cloud.env` |
| Dashboard password (user `sleepsafe`) | HTTP Basic auth on the Vercel site | `scratch\.dashboard_password`; Vercel env `DASHBOARD_PASSWORD` |
| Hotspot key (`SleepSafe-Pi`) | Laptop Mobile Hotspot | `scratch\.hotspot_key`; Windows hotspot settings; Pi netplan file |
| Residence Wi-Fi key | `Theory_Madison_Residence` | Windows saved profile (`netsh wlan show profile name="Theory_Madison_Residence" key=clear`, admin); Pi netplan file. Local copy deleted |
| Pi login password (user `pi`) | Console login and `sudo` | Known to the Pi's owner; not stored by this project |
| SSH private key | Login to the Pi | `C:\Users\harih\.ssh\id_ed25519` (public key in Pi `~/.ssh/authorized_keys`) |
| Vercel CLI session | Deploys / env vars | Vercel CLI auth file in `%APPDATA%\com.vercel.cli\` |
| Databricks browser session | Token management, UI | The Playwright-controlled browser profile (logged in as yahuja2@wisc.edu) |

Token values can't be re-displayed by Databricks; manage/revoke them at **Settings → Developer → Access
tokens**. After the hackathon: revoke all five tokens, delete the `scratch\` secret files, remove the Pi's
`cloud.env`, and rotate the dashboard password.

---

## 14. Runbooks

**Run a real session:** pair/connect the headset → dashboard → Source "Bluetooth headset microphone"
(tick Demo mode if the wearer is awake) → Start → … → Stop → session appears ~3 min later.

**Test without a mic:** dashboard → "Sample recording" → Start (auto-stops after 8 min).

**Replay a file from the laptop:** `scripts\simulate_pi.py <file|chunk dir> --session-id X --job-id 769690655470224 --wait`.

**Local inference:** `.venv\Scripts\python scripts\run_session.py <file|dir> --device cpu [--demo-mode]`.

**Retrain / finalize (CPU, GPU hidden):**
```
set CUDA_VISIBLE_DEVICES=
.venv\Scripts\python scripts\train.py --name v3 --device cpu --batch 32 --steps 3000 --eval-every 250 --patience 4
.venv\Scripts\python scripts\train.py --name v3 --device cpu --finalize-from runs\v3\model_partial.pt
```
Then copy the model to `runs/v2`-style location and run `deploy_databricks.py`.

**Pi unreachable over Wi-Fi:** plug USB-C → `ssh pi@10.55.0.1`; check `networkctl status wlan0`,
`journalctl -u netplan-wpa-wlan0`. For the hotspot fallback: `scratch\hotspot.ps1` on the laptop.

**Dashboard shows "Offline":** `ssh pi@172.16.64.177 systemctl --user restart sleepsafe-pi-agent`.

---

## 15. Open to-dos

1. Re-pair the Px7 S2e and run a real headset session; check the model's behavior on headset sleep audio.
2. Teammate pulls the workspace app changes into his local project.
3. Revoke tokens / clean secrets after the hackathon (section 13).
4. Optional: train v3 beyond step 1000 on a stable machine; add headset-like augmentation (HFP codec,
   noise suppression) or collect a few labeled headset nights.
5. Optional: connect the dashboard's "Ask" to the teammate app (needs OAuth) or a serving endpoint.
