import { SEVERITY } from "@/lib/format";

/** Status color always ships with an icon and a label, never color alone. */
export default function Severity({ value }: { value: string | null | undefined }) {
  const s = value ? SEVERITY[value] : undefined;
  if (!s) return <span className="muted">—</span>;
  return (
    <span className={`badge ${s.status}`}>
      <i aria-hidden="true">{s.icon}</i>
      {s.label}
    </span>
  );
}
