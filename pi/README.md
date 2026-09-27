# SleepSafe on the Raspberry Pi

Pi 4 (Ubuntu 24.04 Server). Capture is Nitesh's recorder (`~/SleepSafe`, branch `nitesh`:
`audio_stream.py` + the `sleepsafe-audio` user service): PipeWire -> FFmpeg (16 kHz mono,
`highpass=f=60,lowpass=f=7000`) -> exact 10 s WAVs -> Files API upload. This folder adds the
**agent** that lets the dashboard start and stop sessions.

```
Dashboard (Vercel) --PUT command.json--> Databricks volume <--poll every 3 s-- Pi agent
Dashboard          <--GET status.json--- Databricks volume <--publish--------- Pi agent
Pi recorder --10 s WAV chunks--> /Volumes/workspace/default/sleepsafe/chunks/<session>/<YYYYMMDD>/
Pi agent (on stop) --> drain spool --> jobs/run-now sleepsafe-process-session --> Delta --> dashboard
```

The Pi only makes outbound HTTPS calls, so it works on any Wi-Fi without port forwarding.

## Files on the Pi

| Path | What |
|---|---|
| `~/sleepsafe-agent/agent.py` | this folder's `agent.py` |
| `~/.config/systemd/user/sleepsafe-pi-agent.service` | runs the agent (enabled, linger on) |
| `~/.config/sleepsafe/cloud.env` (mode 600) | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (files + jobs scopes only), `SLEEPSAFE_VOLUME`, `SLEEPSAFE_JOB_ID`, `DEVICE_ID` |
| `~/sleepsafe-agent/samples/*.wav` | audio for the "Sample recording" source (plays into a PipeWire null sink) |
| `~/.config/systemd/user/sleepsafe-audio.service.d/zz-sleepsafe-session.conf` | written only during a session: switches the recorder to cloud mode and the session's folder; removed on stop (the recorder's own `local-only.conf` is untouched) |

## Sources

- **headset**: the paired Bluetooth headset from `~/SleepSafe/.env` (`AUDIO_SOURCE=bluez_input.…`);
  the agent reconnects it and selects the mSBC (16 kHz) profile. Many headsets apply their own
  noise suppression in headset mode, which can soften breathing sounds.
- **sample**: a WAV from `samples/` played in real time into `sleepsafe_sample`; the recorder captures
  `sleepsafe_sample.monitor`, so the whole capture -> DSP -> upload -> Databricks path runs without a mic.
  The session stops automatically when the sample ends.

## Operate

```bash
systemctl --user status sleepsafe-pi-agent
journalctl --user -u sleepsafe-pi-agent -f
journalctl --user -u sleepsafe-audio -f        # "Captured …" / "Uploaded …" every 10 s while recording
```

## Wi-Fi (Ubuntu 24.04 + Pi 4 Broadcom chip)

`/etc/netplan/60-sleepsafe-wifi.yaml` (root, 600) lists the networks with
`auth: {key-management: psk}`. Netplan's default for a bare `password:` is
`WPA-PSK WPA-PSK-SHA256 SAE` with PMF, which this chip's firmware rejects (association status 16),
so plain WPA2-PSK is set explicitly. `/etc/modprobe.d/sleepsafe-brcmfmac.conf` sets
`options brcmfmac feature_disable=0x82000` (no firmware SAE/handshake offload).
