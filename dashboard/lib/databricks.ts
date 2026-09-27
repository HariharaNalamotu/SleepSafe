import "server-only";
import type { NightReport, SessionRow, WrittenReport } from "./types";

// Server-side only: the token never reaches the browser.
const HOST = (process.env.DATABRICKS_HOST ?? "").replace(/\/$/, "");
const TOKEN = process.env.DATABRICKS_TOKEN ?? "";
const WAREHOUSE = process.env.DATABRICKS_WAREHOUSE_ID ?? "";
const T = process.env.SLEEPSAFE_TABLE_PREFIX ?? "workspace.default.sleepsafe";

type Param = { name: string; value: string };
type Statement = {
  statement_id: string;
  status: { state: string; error?: { message?: string } };
  manifest?: { schema: { columns: { name: string }[] } };
  result?: { data_array?: (string | null)[][] };
};

async function call(path: string, init?: RequestInit): Promise<Statement> {
  const res = await fetch(`${HOST}${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Databricks ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

/** Run a read-only SQL statement on the warehouse; returns rows as objects (values are strings). */
export async function query(statement: string, params: Param[] = []): Promise<Record<string, string | null>[]> {
  if (!HOST || !TOKEN || !WAREHOUSE) throw new Error("Databricks environment variables are not set");
  let r = await call("/api/2.0/sql/statements", {
    method: "POST",
    body: JSON.stringify({
      warehouse_id: WAREHOUSE, statement, parameters: params,
      wait_timeout: "30s", on_wait_timeout: "CONTINUE", format: "JSON_ARRAY", disposition: "INLINE",
    }),
  });
  // A stopped serverless warehouse takes a few seconds to start; poll until the result is ready.
  const deadline = Date.now() + 50_000;
  while ((r.status.state === "PENDING" || r.status.state === "RUNNING") && Date.now() < deadline) {
    await new Promise((ok) => setTimeout(ok, 1500));
    r = await call(`/api/2.0/sql/statements/${r.statement_id}`);
  }
  if (r.status.state !== "SUCCEEDED") {
    throw new Error(`Query ${r.status.state}: ${r.status.error?.message ?? "timed out"}`);
  }
  const cols = r.manifest?.schema.columns.map((c) => c.name) ?? [];
  return (r.result?.data_array ?? []).map((row) => Object.fromEntries(cols.map((c, i) => [c, row[i]])));
}

const num = (v: string | null | undefined) => (v == null || v === "" ? null : Number(v));

export async function listSessions(): Promise<SessionRow[]> {
  const rows = await query(`SELECT * FROM ${T}_v_sessions ORDER BY start_utc DESC LIMIT 200`);
  return rows.map((r) => ({
    session_id: r.session_id ?? "",
    start_utc: r.start_utc,
    end_utc: r.end_utc,
    duration_s: num(r.duration_s),
    valid_audio_s: num(r.valid_audio_s),
    estimated_sleep_s: num(r.estimated_sleep_s),
    estimated_ahi: num(r.estimated_ahi),
    severity_estimate: r.severity_estimate,
    apnea_count: num(r.apnea_count),
    hypopnea_count: num(r.hypopnea_count),
    snore_pct_of_sleep: num(r.snore_pct_of_sleep),
    longest_apnea_s: num(r.longest_apnea_s),
    short_session: r.short_session === "true",
    demo_mode: r.demo_mode === "true",
  }));
}

export async function getSession(id: string): Promise<{ report: NightReport; written: WrittenReport | null } | null> {
  const p = [{ name: "sid", value: id }];
  const [sessions, reports] = await Promise.all([
    query(`SELECT report_json FROM ${T}_sessions WHERE session_id = :sid LIMIT 1`, p),
    query(`SELECT report_markdown, llm_endpoint, created_utc FROM ${T}_reports
           WHERE session_id = :sid ORDER BY created_utc DESC LIMIT 1`, p),
  ]);
  if (!sessions.length || !sessions[0].report_json) return null;
  const report = JSON.parse(sessions[0].report_json) as NightReport;
  const w = reports[0];
  return {
    report,
    written: w?.report_markdown
      ? { markdown: w.report_markdown, llm_endpoint: w.llm_endpoint, created_utc: w.created_utc }
      : null,
  };
}
