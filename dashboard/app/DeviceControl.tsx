"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

type Status = {
  state: string;
  session_id?: string;
  source?: string;
  demo_mode?: boolean;
  started_at?: string | null;
  chunks_uploaded?: number;
  chunks_pending?: number;
  message?: string;
  error?: string | null;
  last_session?: { session_id: string; result: string } | null;
  network?: { ssid: string | null; ip: string | null };
};
type Payload = { status: Status | null; online: boolean; age_s: number | null; error?: string };

const STATE_LABEL: Record<string, string> = {
  idle: "Ready", starting: "Starting…", recording: "Recording", stopping: "Stopping…",
  uploading: "Uploading last chunks…", processing: "Processing on Databricks…", error: "Error",
};

function elapsed(since?: string | null) {
  if (!since) return "";
  const s = Math.max(0, Math.floor((Date.now() - new Date(since).getTime()) / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export default function DeviceControl() {
  const router = useRouter();
  const [data, setData] = useState<Payload | null>(null);
  const [source, setSource] = useState<"headset" | "sample">("headset");
  const [demo, setDemo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [, tick] = useState(0);
  const prevState = useRef<string | undefined>(undefined);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/device", { cache: "no-store" });
      const json = (await res.json()) as Payload;
      setData(json);
      const st = json.status?.state;
      if (prevState.current === "processing" && st === "idle") router.refresh(); // new session in the table
      prevState.current = st;
    } catch {
      /* keep showing the previous status */
    }
  }, [router]);

  useEffect(() => {
    load();
    const poll = setInterval(load, 3000);
    const clock = setInterval(() => tick((n) => n + 1), 1000);
    return () => { clearInterval(poll); clearInterval(clock); };
  }, [load]);

  async function send(action: "start" | "stop") {
    setBusy(true);
    setNote(null);
    try {
      const res = await fetch("/api/device", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, source, demo_mode: demo }),
      });
      const json = await res.json();
      setNote(res.ok ? (action === "start" ? `Start sent (session ${json.queued.session_id}). The recorder picks it up within a few seconds.` : "Stop sent.") : `Failed: ${json.error}`);
      setTimeout(load, 1500);
    } finally {
      setBusy(false);
    }
  }

  const st = data?.status;
  const online = Boolean(data?.online);
  const state = st?.state ?? "unknown";
  const canStart = online && (state === "idle" || state === "error") && !busy;
  const canStop = online && (state === "recording" || state === "starting") && !busy;
  const minutes = st?.chunks_uploaded ? (st.chunks_uploaded * 10) / 60 : 0;

  return (
    <section className="card device" aria-live="polite">
      <div className="device-head">
        <h2 style={{ margin: 0 }}>Recorder <span className="muted" style={{ fontWeight: 400 }}>· Raspberry Pi</span></h2>
        <span className={`badge ${online ? "good" : "critical"}`}>
          <i aria-hidden="true">{online ? "✓" : "✕"}</i>
          {data == null ? "Checking…" : online ? "Online" : `Offline${data.age_s != null ? ` (last seen ${Math.round(data.age_s / 60)} min ago)` : ""}`}
        </span>
      </div>

      <div className="device-grid">
        <div>
          <div className="k">Status</div>
          <div className="device-state">
            {state === "recording" && <span className="rec-dot" aria-hidden="true" />}
            {STATE_LABEL[state] ?? state}
            {state === "recording" && <span className="muted"> · {elapsed(st?.started_at)}</span>}
          </div>
          {st?.message && <div className="muted">{st.message}</div>}
          {st?.error && <div className="error">{st.error}</div>}
        </div>
        <div>
          <div className="k">Session</div>
          <div>{st?.session_id && state !== "idle" ? st.session_id : "—"}</div>
          {(state === "recording" || state === "uploading" || state === "processing") && (
            <div className="muted">
              {st?.chunks_uploaded ?? 0} chunks uploaded ({minutes.toFixed(1)} min){st?.chunks_pending ? ` · ${st.chunks_pending} queued` : ""}
            </div>
          )}
          {state === "idle" && st?.last_session && (
            <div className="muted">
              Last: <Link href={`/sessions/${encodeURIComponent(st.last_session.session_id)}`}>{st.last_session.session_id}</Link>{" "}
              ({st.last_session.result.toLowerCase()})
            </div>
          )}
        </div>
        <div>
          <div className="k">Network</div>
          <div>{st?.network?.ssid ?? "—"}</div>
          <div className="muted">{st?.network?.ip ?? ""}</div>
        </div>
      </div>

      <div className="device-controls">
        <label>
          Source{" "}
          <select value={source} onChange={(e) => setSource(e.target.value as "headset" | "sample")} disabled={!canStart}>
            <option value="headset">Bluetooth headset microphone</option>
            <option value="sample">Sample recording (PSG apnea clip, 8 min)</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={demo} onChange={(e) => setDemo(e.target.checked)} disabled={!canStart} /> Demo mode (count events while awake)
        </label>
        <div className="buttons">
          <button className="btn primary" onClick={() => send("start")} disabled={!canStart}>Start recording</button>
          <button className="btn" onClick={() => send("stop")} disabled={!canStop}>Stop</button>
        </div>
      </div>
      {note && <p className="muted" style={{ margin: "8px 0 0" }}>{note}</p>}
    </section>
  );
}
