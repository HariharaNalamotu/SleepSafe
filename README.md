# SleepSafe Raspberry Pi audio capture

This branch contains a Python 3 + FFmpeg pipeline for Ubuntu on a Raspberry Pi:

```text
Bluetooth headset mic (HFP/HSP)
  -> BlueZ + PipeWire's PulseAudio interface
  -> FFmpeg: resample/mix to 16 kHz mono, gentle filters, signed 16-bit PCM
  -> Python: exact 160,000-sample / 10-second WAV files on local disk
  -> background HTTPS upload to a Databricks Unity Catalog volume
  -> optional Auto Loader stream into a Delta table
```

The Pi uploads a completed file every ten seconds when capture and networking are
healthy. This is micro-batch ingestion: cloud availability is ten seconds plus
upload time; Databricks processing adds its own scheduling/compute delay. It is
not a live audio socket or a model-serving endpoint. No Python pip dependencies
are required. Ubuntu 24.04 LTS with PipeWire is the setup baseline below; other
Ubuntu versions may need different audio-service configuration.

## 1. Install audio support on the Pi

Run as the Linux user that will record audio (use sudo only where shown):

```bash
sudo apt update
sudo apt install bluez pipewire pipewire-pulse wireplumber libspa-0.2-bluetooth pulseaudio-utils ffmpeg python3
sudo systemctl enable --now bluetooth
systemctl --user enable --now pipewire.socket pipewire-pulse.socket wireplumber.service
pactl info
```

`pactl info` should report PulseAudio on PipeWire. If an older Ubuntu installation
already runs a standalone PulseAudio server, resolve that service conflict before
starting PipeWire. FFmpeg must list `pulse` under `ffmpeg -devices`.

## 2. Pair and select the headset microphone

Put the headset in pairing mode, then use the interactive Bluetooth console:

```text
bluetoothctl
power on
agent on
default-agent
scan on
pair AA:BB:CC:DD:EE:FF
trust AA:BB:CC:DD:EE:FF
connect AA:BB:CC:DD:EE:FF
scan off
quit
```

Replace the address with the headset's address. Confirm any pairing prompt.
Disconnect the headset from your phone if it keeps taking the microphone connection.

```bash
pactl list cards
# Choose an available HFP/HSP profile listed for YOUR card; example only:
pactl set-card-profile bluez_card.AA_BB_CC_DD_EE_FF headset-head-unit
pactl list short sources
```

Profile names vary: some systems expose a profile such as
`headset-head-unit-msbc`. Choose mSBC/wideband when available. Copy the exact
Bluetooth **input source name** from the final command; do not choose a `.monitor`
source. A2DP playback does not provide the classic Bluetooth headset microphone.
HFP with mSBC typically supplies 16 kHz audio; CVSD typically supplies 8 kHz.
Converting an 8 kHz source to 16 kHz does not recover missing frequencies. A
headset's internal noise suppression may also alter quiet breathing sounds.

The program opens your explicit source name and does not silently fall back to
another microphone. Pairing and profile selection belong to the Ubuntu audio
stack; the Python code consumes the source it exposes.

## 3. Verify local recording first

From this repository on the Pi:

```bash
python3 audio_stream.py --source YOUR_EXACT_SOURCE_NAME --local-only
```

Speak into the microphone, wait at least 20 seconds, then press Ctrl+C. Inspect
and listen to one of the generated WAV files:

```bash
ffprobe spool/YOUR_CHUNK.wav
ffplay -autoexit spool/YOUR_CHUNK.wav
```

Each file is mono, 16,000 Hz, 16-bit PCM, and exactly 10 seconds (~320 KB).
Default processing is a 60 Hz high-pass and a 7 kHz low-pass, after resampling.
It is deliberately mild: no automatic gain normalization, gating, or noise
subtraction that might erase quiet sounds. Use `--filters anull` for a baseline,
or supply an FFmpeg filter chain via `--filters` / `AUDIO_FILTERS`. Evaluate
filter choices against your downstream model and actual headset recordings.

## 4. Configure Databricks uploads

Create a Unity Catalog schema/volume in your workspace, for example using SQL
(adapt the existing catalog name):

```sql
CREATE SCHEMA IF NOT EXISTS main.sleepsafe;
CREATE VOLUME IF NOT EXISTS main.sleepsafe.audio;
```

The uploader identity needs `USE CATALOG`, `USE SCHEMA`, and `WRITE VOLUME` on
the chosen objects. The notebook reader additionally needs `READ VOLUME` and
permission to create/write its Delta table and checkpoint. Use a workspace
access token permitted by your workspace policy. This prototype accepts a bearer
token through the environment; it does not implement OAuth token refresh. For
an unattended deployment, add managed service-principal OAuth credential renewal
or arrange token rotation before expiry.

```bash
cp .env.example .env
chmod 600 .env
# Edit .env: source, device ID, spool directory, workspace URL, token, volume.
# Values are shell-compatible; quote values if you introduce spaces.
set -a
. ./.env
set +a
python3 audio_stream.py
```

Keep `.env` out of Git. The host is the HTTPS workspace origin, without an API
path. The destination must be `/Volumes/<catalog>/<schema>/<volume>[/folder]`.
Files go into UTC date folders below it. Parent directories are created through
the Files API. Filenames contain the UTC **chunk completion time**, device ID,
and a UUID. Keep the Pi clock synchronized; timestamps are not hardware-precise.

The uploader sends raw WAV bytes using `PUT /api/2.0/fs/files/...`. It retries the
same filename with overwrite enabled so a lost acknowledgment does not create
a second file. It deletes a local WAV only after a successful server response.
Failed uploads back off from 1 to 60 seconds; HTTP status codes appear in logs.
For 401/403 check credentials/permissions; for 404 check the host and volume.

Pending local-test files in the same spool will upload when cloud mode starts.
Use a separate `--spool` directory for recordings you want to keep local. A spool
belongs to one destination: drain it before changing the Databricks configuration.

## 5. Optional Databricks ingestion

Import [databricks/ingest_audio.py](databricks/ingest_audio.py) as a notebook,
adjust `SOURCE`, `CHECKPOINT`, and `TABLE`, and run it on supported Databricks
compute. It uses Auto Loader `binaryFile` ingestion and stores each WAV's bytes
and identifying columns in a Delta table. Decode the WAV header before model
inference. The ten-second trigger requests a processing cadence, not a latency
guarantee. Keep the checkpoint outside the incoming folder and reuse it across
restarts. The example keeps a stream running and therefore requires running
compute; batch/scheduled processing is also possible.

## 6. Run after login / boot

The service is a **user service** so it can access that user's PipeWire session.
The template assumes the repository is at `~/SleepSafe` and `.env` lives there:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/sleepsafe-audio.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sleepsafe-audio
journalctl --user -u sleepsafe-audio -f
# Optional: keep the user service running after logout and start it at boot.
sudo loginctl enable-linger "$USER"
```

For a headless Pi, WirePlumber's Bluetooth seat monitoring may prevent an inactive
user from owning the headset. First check `journalctl --user -u wireplumber` and
`pactl list short sources` under the service's user. WirePlumber 0.5 supports a
`~/.config/wireplumber/wireplumber.conf.d/51-headless.conf` override:

```text
wireplumber.profiles = {
  main = {
    monitor.bluez.seat-monitoring = disabled
  }
}
```

Check `wireplumber --version` first. For WirePlumber 0.4 (as shipped by Ubuntu
24.04), use `~/.config/wireplumber/bluetooth.lua.d/80-disable-logind.lua` instead:

```lua
bluez_monitor.properties["with-logind"] = false
```

Create the parent directory if needed. Apply only the snippet matching your
version, then `systemctl --user restart wireplumber`. See the official
[0.4/0.5 migration guide](https://pipewire.pages.freedesktop.org/wireplumber/daemon/configuration/migration.html)
for these equivalent settings. Linger alone does not guarantee Bluetooth access. Verify
capture after an actual reboot without a desktop login. The recorder retries
failed/stalled capture, but does not re-pair or force reconnection; use
`bluetoothctl connect ADDRESS` and recheck the profile if the headset stays offline.

## Reliability and limits

- Capture and upload operate independently. Pending WAV files survive process
  restarts; `.part` files from interrupted disk writes are never uploaded and can
  be inspected/removed manually. Only one process may own a spool directory.
- Default queue capacity is 512 MiB (about 4.7 hours at 32 KB/s). At capacity,
  capture pauses, keeping existing chunks. Audio during the pause is lost; logs
  report this. Daily raw upload volume is about 2.76 GB per continuously running Pi.
- Disconnects, shutdown, or capture failures discard the incomplete in-memory
  chunk (normally less than ten seconds). Reconnection starts a new chunk; no
  synthetic silence is inserted. WAVs are individually complete, but the stream
  is not guaranteed gap-free. A hung source is restarted after 30 seconds without
  data; a source emitting silence cannot be distinguished from a quiet room.
- This first implementation is intended for a small number of Pis. At fleet
  scale, consider direct object-storage ingestion and fewer/larger objects.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests use synthetic FFmpeg audio and mocked HTTP responses, covering resampling,
mono output, exact chunk lengths, partial tails, durable queue behavior, request
construction, retry backoff, and configuration validation. Bluetooth and actual
Databricks access must be verified on your Pi/workspace.

References: [WirePlumber Bluetooth configuration](https://pipewire.pages.freedesktop.org/wireplumber/daemon/configuration/bluetooth.html),
[FFmpeg audio filters](https://ffmpeg.org/ffmpeg-filters.html),
[Databricks Files API](https://docs.databricks.com/api/files/v2/file),
[Auto Loader options](https://docs.databricks.com/aws/en/spark/api-options).
