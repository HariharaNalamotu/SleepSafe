import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getSession } from "@/lib/databricks";
import { TYPE_LABEL, fmtClock, fmtDate, fmtDuration } from "@/lib/format";
import Severity from "../../Severity";
import HourlyBars from "./HourlyBars";
import Timeline from "./Timeline";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const TYPE_COLOR: Record<string, string> = {
  apnea: "var(--series-1)", hypopnea: "var(--series-2)", snore_episode: "var(--series-3)",
  cough: "var(--series-4)", gasp: "var(--series-4)", snort: "var(--series-4)", wheeze: "var(--series-4)",
  speech: "var(--series-4)", wake_period: "var(--context)", low_signal: "var(--context)",
};

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let data;
  try {
    data = await getSession(decodeURIComponent(id));
  } catch (e) {
    return (
      <div className="card">
        <p className="error">Could not load this session from Databricks: {(e as Error).message}</p>
      </div>
    );
  }
  if (!data) notFound();
  const { report, written } = data;
  const { recording: rec, summary: sum } = report;
  const events = report.events.filter((e) => e.type !== "low_signal" || e.duration_s > 0);
  const hourly = sum.events_per_hour_by_hour ?? [];
  const validPct = rec.duration_s ? Math.round((100 * rec.valid_audio_s) / rec.duration_s) : null;

  return (
    <>
      <p style={{ margin: "0 0 8px" }}><Link href="/">← All sessions</Link></p>
      <div style={{ marginBottom: 16 }}>
        <h1>
          {report.session_id}
          {report.model?.demo_mode && <span className="flag">demo mode</span>}
          {sum.short_session && <span className="flag">short session</span>}
        </h1>
        <p className="sub">
          {fmtDate(rec.start_utc)} · {fmtClock(rec.start_utc)}–{fmtClock(rec.end_utc)} UTC · {fmtDuration(rec.duration_s)} recorded
          {validPct != null && ` · ${validPct}% usable audio`}
        </p>
      </div>

      <section className="tiles" aria-label="Summary">
        <div className="tile hero">
          <div className="k">Breathing events per hour</div>
          <div className="v">{sum.estimated_ahi ?? "—"}</div>
          <div className="note"><Severity value={sum.severity_estimate} /></div>
        </div>
        <div className="tile">
          <div className="k">Apneas</div>
          <div className="v">{sum.counts.apnea ?? 0}</div>
          <div className="note">longest {sum.longest_apnea_s} s</div>
        </div>
        <div className="tile">
          <div className="k">Hypopneas</div>
          <div className="v">{sum.counts.hypopnea ?? 0}</div>
          <div className="note">reduced breathing</div>
        </div>
        <div className="tile">
          <div className="k">Estimated sleep</div>
          <div className="v">{fmtDuration(rec.estimated_sleep_s)}</div>
          <div className="note">of {fmtDuration(rec.duration_s)} recorded</div>
        </div>
        <div className="tile">
          <div className="k">Snoring</div>
          <div className="v">{sum.snore_pct_of_sleep ?? "—"}<small>{sum.snore_pct_of_sleep != null && " % of sleep"}</small></div>
          <div className="note">{sum.counts.snore_episode ?? 0} episodes</div>
        </div>
      </section>

      <section className="card">
        <h2>Event timeline</h2>
        <p className="sub" style={{ marginBottom: 12 }}>
          Every detected event across the session. Hover or tab to a mark for details. Gray lanes are context.
        </p>
        <Timeline events={events} duration={rec.duration_s} startUtc={rec.start_utc} />
      </section>

      {hourly.length > 1 && (
        <section className="card">
          <h2>Breathing events per hour of sleep, by hour</h2>
          <HourlyBars values={hourly} startUtc={rec.start_utc} />
        </section>
      )}

      <section className="card">
        <h2>Events ({events.length})</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Start (UTC)</th>
                <th>End (UTC)</th>
                <th>Type</th>
                <th className="num">Duration</th>
                <th className="num">Confidence</th>
                <th>Evidence</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={e.id ?? i}>
                  <td>{fmtClock(e.start_utc, true)}</td>
                  <td>{fmtClock(e.end_utc, true)}</td>
                  <td>
                    <span className="swatch" style={{ background: TYPE_COLOR[e.type] ?? "var(--context)" }} aria-hidden="true" />
                    {TYPE_LABEL[e.type] ?? e.type}
                  </td>
                  <td className="num">{e.duration_s} s</td>
                  <td className="num">{e.confidence != null ? e.confidence.toFixed(2) : "—"}</td>
                  <td className="muted">
                    {e.confidence_basis === "zero_shot" ? "pretrained sound classifier" : e.confidence_basis === "trained" ? "trained model" : e.reason ?? ""}
                    {e.evidence?.terminated_by && ` · ended with a ${e.evidence.terminated_by}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Written report</h2>
        {written ? (
          <>
            <div className="report">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{written.markdown}</ReactMarkdown>
            </div>
            <p className="muted" style={{ marginTop: 12 }}>
              Generated by {written.llm_endpoint ?? "an LLM"} from the model output{written.created_utc && ` at ${fmtClock(written.created_utc)} UTC`}.
            </p>
          </>
        ) : (
          <p className="muted">No written report for this session yet.</p>
        )}
      </section>

      <section className="card">
        <h2>Limitations</h2>
        <ul className="caveats">
          {report.caveats.map((c) => <li key={c}>{c}</li>)}
        </ul>
      </section>
    </>
  );
}
