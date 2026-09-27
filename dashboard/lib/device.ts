import "server-only";

// Recorder control through the Databricks volume: the dashboard writes command.json, the Pi agent
// polls it and publishes status.json. Needs a token with the "files" scope. Server-side only.
const HOST = (process.env.DATABRICKS_HOST ?? "").replace(/\/$/, "");
const TOKEN = process.env.DATABRICKS_TOKEN ?? "";
const VOLUME = process.env.SLEEPSAFE_VOLUME ?? "/Volumes/workspace/default/sleepsafe";
export const DEVICE_ID = process.env.SLEEPSAFE_DEVICE_ID ?? "pi-01";
const OFFLINE_AFTER_S = 30;

const filesUrl = (name: string) =>
  `${HOST}/api/2.0/fs/files${encodeURI(`${VOLUME}/control/${DEVICE_ID}/${name}`)}`;

export type DeviceStatus = {
  device_id: string;
  state: "idle" | "starting" | "recording" | "stopping" | "uploading" | "processing" | "error";
  session_id?: string;
  source?: "headset" | "sample";
  demo_mode?: boolean;
  started_at?: string | null;
  chunks_uploaded?: number;
  chunks_pending?: number;
  capture_service?: string;
  job_run_id?: number | null;
  message?: string;
  error?: string | null;
  last_session?: { session_id: string; result: string; finished_at: string } | null;
  network?: { ssid: string | null; ip: string | null };
  updated_at?: string;
};

export async function getDeviceStatus(): Promise<{ status: DeviceStatus | null; online: boolean; age_s: number | null }> {
  const res = await fetch(filesUrl("status.json"), { headers: { Authorization: `Bearer ${TOKEN}` }, cache: "no-store" });
  if (res.status === 404) return { status: null, online: false, age_s: null };
  if (!res.ok) throw new Error(`Databricks ${res.status}`);
  const status = (await res.json()) as DeviceStatus;
  const age = status.updated_at ? (Date.now() - new Date(status.updated_at).getTime()) / 1000 : null;
  return { status, online: age != null && age < OFFLINE_AFTER_S, age_s: age == null ? null : Math.round(age) };
}

export type Command = {
  id: string;
  action: "start" | "stop";
  session_id?: string;
  source?: "headset" | "sample";
  demo_mode?: boolean;
  requested_at: string;
};

export async function sendCommand(cmd: Command): Promise<void> {
  const res = await fetch(`${filesUrl("command.json")}?overwrite=true`, {
    method: "PUT",
    headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/octet-stream" },
    body: JSON.stringify(cmd),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Databricks ${res.status}: ${(await res.text()).slice(0, 200)}`);
}
