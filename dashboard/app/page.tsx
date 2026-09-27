import Link from "next/link";
import { listSessions } from "@/lib/databricks";
import { fmtClock, fmtDate, fmtDuration } from "@/lib/format";
import Severity from "./Severity";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

export default async function Home() {
  let sessions;
  try {
    sessions = await listSessions();
  } catch (e) {
    return (
      <div className="card">
        <h1>Sessions</h1>
        <p className="error">Could not load sessions from Databricks: {(e as Error).message}</p>
      </div>
    );
  }

  return (
    <>
      <div style={{ marginBottom: 16 }}>
        <h1>Sessions</h1>
        <p className="sub">
          {sessions.length} recorded session{sessions.length === 1 ? "" : "s"}, newest first. Each session is
          processed when recording stops.
        </p>
      </div>
      <div className="card">
        {sessions.length === 0 ? (
          <p className="muted">No sessions yet. They appear here a few minutes after a recording ends.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Session</th>
                  <th>Date</th>
                  <th>Start (UTC)</th>
                  <th className="num">Duration</th>
                  <th className="num">Events / hour</th>
                  <th>Severity estimate</th>
                  <th className="num">Apneas</th>
                  <th className="num">Hypopneas</th>
                  <th className="num">Snoring</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => (
                  <tr key={s.session_id}>
                    <td>
                      <Link href={`/sessions/${encodeURIComponent(s.session_id)}`}>{s.session_id}</Link>
                      {s.demo_mode && <span className="flag">demo</span>}
                    </td>
                    <td>{fmtDate(s.start_utc)}</td>
                    <td>{fmtClock(s.start_utc)}</td>
                    <td className="num">{fmtDuration(s.duration_s)}</td>
                    <td className="num">{s.estimated_ahi ?? "—"}</td>
                    <td><Severity value={s.severity_estimate} /></td>
                    <td className="num">{s.apnea_count ?? "—"}</td>
                    <td className="num">{s.hypopnea_count ?? "—"}</td>
                    <td className="num">{s.snore_pct_of_sleep == null ? "—" : `${s.snore_pct_of_sleep}%`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
