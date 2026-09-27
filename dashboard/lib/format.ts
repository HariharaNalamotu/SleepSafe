export const SEVERITY: Record<string, { status: "good" | "warning" | "serious" | "critical"; icon: string; label: string }> = {
  normal: { status: "good", icon: "✓", label: "Normal" },
  mild: { status: "warning", icon: "!", label: "Mild" },
  moderate: { status: "serious", icon: "!!", label: "Moderate" },
  severe: { status: "critical", icon: "✕", label: "Severe" },
};

export function fmtDuration(s: number | null | undefined): string {
  if (s == null) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
}

/** Clock time in UTC: the reports carry UTC timestamps and no user timezone. */
export function fmtClock(iso: string | null | undefined, seconds = false): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toISOString().slice(11, seconds ? 19 : 16);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

export const TYPE_LABEL: Record<string, string> = {
  apnea: "Apnea",
  hypopnea: "Hypopnea",
  snore_episode: "Snoring",
  wake_period: "Awake",
  low_signal: "No signal",
  cough: "Cough",
  gasp: "Gasp",
  snort: "Snort",
  wheeze: "Wheeze",
  speech: "Speech",
};
