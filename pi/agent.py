#!/usr/bin/env python3
"""SleepSafe Pi agent: dashboard-controlled recording sessions, with Databricks as the go-between.

The dashboard writes   <volume>/control/<device>/command.json   {"id", "action": "start"|"stop", ...}
This agent polls it every few seconds and publishes
                        <volume>/control/<device>/status.json
so the Pi needs only outbound HTTPS (no inbound ports, works on any Wi-Fi).

Recording reuses the existing capture service (sleepsafe-audio: PipeWire -> FFmpeg filters ->
10 s WAV spool -> Files API upload) unchanged; a systemd drop-in points it at the session's
volume folder. On stop the agent drains the spool, then starts the Databricks processing job.

Sources: "headset" (paired Bluetooth headset, HFP mSBC) or "sample" (a WAV played into a
PipeWire null sink, so the full capture/DSP/upload path runs without a microphone).

Config: ~/.config/sleepsafe/cloud.env (DATABRICKS_HOST, DATABRICKS_TOKEN, SLEEPSAFE_VOLUME,
SLEEPSAFE_JOB_ID, DEVICE_ID). Python standard library only.
"""
import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

HOME = Path.home()
CLOUD_ENV = HOME / ".config/sleepsafe/cloud.env"
STATE_DIR = HOME / ".local/state/sleepsafe"
SESSION_ENV = STATE_DIR / "session.env"
AGENT_STATE = STATE_DIR / "agent.json"
DROPIN = HOME / ".config/systemd/user/sleepsafe-audio.service.d/zz-sleepsafe-session.conf"
SPOOL_ROOT = HOME / "SleepSafe/spool-sessions"
SAMPLES = HOME / "sleepsafe-agent/samples"
CAPTURE_ENV = HOME / "SleepSafe/.env"
NULL_SINK = "sleepsafe_sample"
POLL_S, HEARTBEAT_S = 3, 10
VERSION = "1.0"
LOG = logging.getLogger("sleepsafe-agent")


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sh(*cmd, check=True, timeout=60) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {(r.stderr or r.stdout).strip()[:200]}")
    return r.stdout


class Databricks:
    def __init__(self, host: str, token: str):
        self.host, self.token = host.rstrip("/"), token

    def _req(self, method, path, data=None, ctype="application/octet-stream", timeout=30):
        req = Request(self.host + path, data=data, method=method,
                      headers={"Authorization": f"Bearer {self.token}", "Content-Type": ctype})
        with urlopen(req, timeout=timeout) as r:
            return r.read()

    def get_file(self, vol_path):
        try:
            return self._req("GET", "/api/2.0/fs/files" + quote(vol_path))
        except HTTPError as e:
            if e.code == 404:
                return None
            raise

    def put_file(self, vol_path, data: bytes):
        self._req("PUT", "/api/2.0/fs/files" + quote(vol_path) + "?overwrite=true", data)

    def mkdir(self, vol_path):
        self._req("PUT", "/api/2.0/fs/directories" + quote(vol_path))

    def count_wavs(self, vol_dir) -> int:
        """WAV files under vol_dir/<date>/ (the recorder's layout)."""
        try:
            top = json.loads(self._req("GET", "/api/2.0/fs/directories" + quote(vol_dir)) or b"{}")
        except HTTPError as e:
            if e.code == 404:
                return 0
            raise
        n = 0
        for c in top.get("contents", []):
            if c.get("is_directory"):
                sub = json.loads(self._req("GET", "/api/2.0/fs/directories" + quote(c["path"])) or b"{}")
                n += sum(1 for f in sub.get("contents", []) if f["name"].endswith(".wav"))
        return n

    def run_job(self, job_id, params) -> int:
        body = json.dumps({"job_id": int(job_id), "job_parameters": params}).encode()
        return json.loads(self._req("POST", "/api/2.2/jobs/run-now", body, "application/json"))["run_id"]

    def run_state(self, run_id) -> dict:
        return json.loads(self._req("GET", f"/api/2.2/jobs/runs/get?run_id={run_id}"))["state"]


class Agent:
    def __init__(self):
        cfg = load_env(CLOUD_ENV)
        self.cfg = cfg
        self.db = Databricks(cfg["DATABRICKS_HOST"], cfg["DATABRICKS_TOKEN"])
        self.device = cfg.get("DEVICE_ID", "pi-01")
        self.volume = cfg["SLEEPSAFE_VOLUME"].rstrip("/")
        self.control = f"{self.volume}/control/{self.device}"
        self.job_id = cfg["SLEEPSAFE_JOB_ID"]
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        saved = json.loads(AGENT_STATE.read_text()) if AGENT_STATE.exists() else {}
        self.last_command_id = saved.get("last_command_id")
        self.status = {"device_id": self.device, "state": "idle", "agent_version": VERSION,
                       "last_session": saved.get("last_session")}
        self.player = None
        self.session = None
        self.last_heartbeat = 0.0
        if self.last_command_id is None:  # first run: don't replay whatever command is already there
            cmd = self.read_command()
            self.remember(cmd["id"] if cmd else "none")

    # ---------- bookkeeping ----------
    def remember(self, command_id):
        self.last_command_id = command_id
        AGENT_STATE.write_text(json.dumps({"last_command_id": command_id,
                                           "last_session": self.status.get("last_session")}))

    def read_command(self):
        raw = self.db.get_file(f"{self.control}/command.json")
        return json.loads(raw) if raw else None

    def publish(self, **changes):
        self.status.update(changes)
        self.status["updated_at"] = now()
        self.status["network"] = self.network()
        try:
            self.db.put_file(f"{self.control}/status.json", json.dumps(self.status).encode())
            self.last_heartbeat = time.monotonic()
        except (HTTPError, URLError, OSError) as e:
            LOG.warning("status publish failed: %s", e)

    @staticmethod
    def network():
        ssid = None
        try:
            out = subprocess.run(["networkctl", "status", "wlan0", "--no-pager"],
                                 capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"(?:Wi-?Fi access point|SSID):\s*(.+?)(?:\s+\([0-9a-f:]{17}\))?$", out, re.M | re.I)
            ssid = m.group(1).strip() if m else None
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("1.1.1.1", 80))
            ip = s.getsockname()[0]
            s.close()
        except OSError:
            ip = None
        return {"ssid": ssid, "ip": ip}

    # ---------- audio sources ----------
    def headset_source(self) -> str:
        source = load_env(CAPTURE_ENV).get("AUDIO_SOURCE", "")
        m = re.match(r"bluez_input\.([0-9A-F_]{17})", source)
        if not m:
            raise RuntimeError("No Bluetooth headset source configured in ~/SleepSafe/.env")
        mac = m.group(1).replace("_", ":")
        sh("bluetoothctl", "--timeout", "20", "connect", mac, check=False, timeout=30)
        card = "bluez_card." + m.group(1)
        for _ in range(20):
            if subprocess.run(["pactl", "set-card-profile", card, "headset-head-unit-msbc"],
                              capture_output=True).returncode == 0:
                break
            time.sleep(1)
        if source not in sh("pactl", "list", "short", "sources"):
            raise RuntimeError("Headset microphone not available: turn the headset on and pair it")
        return source

    def sample_source(self) -> str:
        if NULL_SINK not in sh("pactl", "list", "short", "sinks"):
            sh("pactl", "load-module", "module-null-sink", f"sink_name={NULL_SINK}",
               "sink_properties=device.description=SleepSafeSample")
        return f"{NULL_SINK}.monitor"

    def sample_file(self, name: str | None) -> Path:
        files = sorted(SAMPLES.glob("*.wav"))
        if not files:
            raise RuntimeError(f"No sample WAVs in {SAMPLES}")
        return next((f for f in files if f.name == name), files[0])

    # ---------- session control ----------
    def start(self, cmd):
        if self.status["state"] not in ("idle", "error"):
            self.publish(message=f"Ignored start: device is {self.status['state']}")
            return
        sid = cmd.get("session_id") or datetime.now(timezone.utc).strftime("pi-%Y%m%d-%H%M%S")
        source = cmd.get("source", "headset")
        demo = bool(cmd.get("demo_mode", False))
        self.publish(state="starting", session_id=sid, source=source, demo_mode=demo,
                     message="Preparing audio source", job_run_id=None, chunks_uploaded=0,
                     chunks_pending=0, started_at=None, error=None)
        audio_source = self.headset_source() if source == "headset" else self.sample_source()
        spool = SPOOL_ROOT / sid
        spool.mkdir(parents=True, exist_ok=True)
        vol_dir = f"{self.volume}/chunks/{sid}"
        self.db.mkdir(vol_dir)
        started = now()
        self.db.put_file(f"{vol_dir}/session.json", json.dumps(
            {"session_id": sid, "start_utc": started, "device_id": self.device, "source": source}).encode())
        SESSION_ENV.write_text(
            f"DATABRICKS_VOLUME={vol_dir}\nSPOOL_DIR={spool}\nAUDIO_SOURCE={audio_source}\nDEVICE_ID={self.device}\n")
        DROPIN.parent.mkdir(parents=True, exist_ok=True)
        DROPIN.write_text(
            "[Service]\nExecStart=\nExecStart=/usr/bin/python3 %h/SleepSafe/audio_stream.py\n"
            f"EnvironmentFile={CLOUD_ENV}\nEnvironmentFile={SESSION_ENV}\n")
        sh("systemctl", "--user", "daemon-reload")
        sh("systemctl", "--user", "restart", "sleepsafe-audio")
        if source == "sample":
            f = self.sample_file(cmd.get("sample"))
            time.sleep(2)  # let capture attach before audio starts
            self.player = subprocess.Popen(["paplay", f"--device={NULL_SINK}", str(f)])
            msg = f"Streaming sample {f.name} through the capture pipeline"
        else:
            msg = "Recording from headset"
        self.session = {"id": sid, "spool": spool, "vol_dir": vol_dir, "demo": demo}
        self.publish(state="recording", started_at=started, message=msg)

    def stop(self, reason="Stopped from dashboard"):
        if not self.session:
            self.publish(message="Ignored stop: not recording")
            return
        s = self.session
        self.publish(state="stopping", message=reason)
        if self.player and self.player.poll() is None:
            self.player.terminate()
        self.player = None
        subprocess.run(["systemctl", "--user", "stop", "sleepsafe-audio"], timeout=120)
        # Drain whatever the recorder had not uploaded yet (same layout: <vol_dir>/<YYYYMMDD>/<name>).
        pending = sorted(s["spool"].glob("*.wav"))
        self.publish(state="uploading", chunks_pending=len(pending), message=f"Uploading last {len(pending)} chunks")
        for p in pending:
            d = f"{s['vol_dir']}/{p.name[:8]}"
            self.db.mkdir(d)
            self.db.put_file(f"{d}/{p.name}", p.read_bytes())
            p.unlink()
        DROPIN.unlink(missing_ok=True)
        sh("systemctl", "--user", "daemon-reload")
        uploaded = self.db.count_wavs(s["vol_dir"])
        run_id = self.db.run_job(self.job_id, {"session_id": s["id"], "demo_mode": str(s["demo"]).lower()})
        self.session = None
        self.publish(state="processing", job_run_id=run_id, chunks_uploaded=uploaded, chunks_pending=0,
                     message=f"Uploaded {uploaded} chunks; Databricks job running")

    def tick(self):
        """Periodic work: session progress, auto-stop at sample end, job completion."""
        st = self.status["state"]
        if st == "recording" and self.session:
            if self.player and self.player.poll() is not None:
                self.stop("Sample finished")
                return
            pending = len(list(self.session["spool"].glob("*.wav")))
            try:
                uploaded = self.db.count_wavs(self.session["vol_dir"])
            except (HTTPError, URLError, OSError):
                uploaded = self.status.get("chunks_uploaded", 0)
            active = subprocess.run(["systemctl", "--user", "is-active", "sleepsafe-audio"],
                                    capture_output=True, text=True).stdout.strip()
            self.publish(chunks_uploaded=uploaded, chunks_pending=pending, capture_service=active)
        elif st == "processing" and self.status.get("job_run_id"):
            state = self.db.run_state(self.status["job_run_id"])
            if state["life_cycle_state"] in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                result = state.get("result_state", state["life_cycle_state"])
                last = {"session_id": self.status["session_id"], "result": result, "finished_at": now()}
                self.status["last_session"] = last
                self.remember(self.last_command_id)
                self.publish(state="idle", message=f"Session {last['session_id']} processed: {result}")
        elif time.monotonic() - self.last_heartbeat >= HEARTBEAT_S:
            self.publish()

    def run(self):
        self.publish(state="idle", message="Agent started")
        while True:
            try:
                cmd = self.read_command()
                if cmd and cmd.get("id") != self.last_command_id:
                    self.remember(cmd["id"])
                    LOG.info("command %s: %s", cmd["id"], cmd.get("action"))
                    if cmd.get("action") == "start":
                        self.start(cmd)
                    elif cmd.get("action") == "stop":
                        self.stop()
                self.tick()
            except Exception as e:  # keep the agent alive; surface the problem on the dashboard
                LOG.exception("agent error")
                if self.session:
                    subprocess.run(["systemctl", "--user", "stop", "sleepsafe-audio"], timeout=120)
                    DROPIN.unlink(missing_ok=True)
                    subprocess.run(["systemctl", "--user", "daemon-reload"])
                    self.session = None
                self.publish(state="error", error=str(e)[:300], message="Error — see error field")
            time.sleep(POLL_S)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Agent().run()
