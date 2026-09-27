"use client";

import { useEffect, useRef, useState } from "react";
import type { NightEvent } from "@/lib/types";
import { TYPE_LABEL } from "@/lib/format";

// Lanes in fixed order; colors follow the event type, never its rank.
const LANES = [
  { key: "apnea", label: "Apnea", types: ["apnea"], color: "var(--series-1)" },
  { key: "hypopnea", label: "Hypopnea", types: ["hypopnea"], color: "var(--series-2)" },
  { key: "snore", label: "Snoring", types: ["snore_episode"], color: "var(--series-3)" },
  { key: "sounds", label: "Other sounds", types: ["cough", "gasp", "snort", "wheeze", "speech"], color: "var(--series-4)" },
  { key: "wake", label: "Awake", types: ["wake_period"], color: "var(--context)" },
  { key: "signal", label: "No signal", types: ["low_signal"], color: "var(--context)" },
];

const LABEL_W_WIDE = 116;
const LABEL_W_NARROW = 88;
const RIGHT = 12;
const LANE_H = 30;
const BAR_H = 14;
const AXIS_H = 26;
const MIN_HIT = 24;

const MIN_TICK_PX = 52; // keep axis labels from colliding on narrow screens

function tickStep(duration: number, plotW: number) {
  const steps = [300, 600, 900, 1800, 3600, 7200, 10800];
  return steps.find((s) => (s / duration) * plotW >= MIN_TICK_PX) ?? steps[steps.length - 1];
}

function clock(startMs: number, offset: number, seconds = false) {
  return new Date(startMs + offset * 1000).toISOString().slice(11, seconds ? 19 : 16);
}

type Tip = { x: number; y: number; e: NightEvent };

export default function Timeline({ events, duration, startUtc }: { events: NightEvent[]; duration: number; startUtc: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [tip, setTip] = useState<Tip | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(240, Math.floor(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const startMs = new Date(startUtc).getTime();
  const LABEL_W = width < 480 ? LABEL_W_NARROW : LABEL_W_WIDE;
  const plotW = width - LABEL_W - RIGHT;
  const x = (t: number) => LABEL_W + (Math.min(Math.max(t, 0), duration) / duration) * plotW;
  const height = LANES.length * LANE_H + AXIS_H;
  const step = tickStep(duration, plotW);
  const ticks: number[] = [];
  for (let t = 0; t <= duration; t += step) ticks.push(t);

  const show = (e: NightEvent, lane: number) =>
    setTip({ x: (x(e.start_offset_s) + x(e.end_offset_s)) / 2, y: lane * LANE_H + (LANE_H - BAR_H) / 2, e });

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Event timeline by type across the session">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={LANES.length * LANE_H} stroke="var(--grid)" strokeWidth={1} />
            <text x={x(t)} y={LANES.length * LANE_H + 17} textAnchor={t === 0 ? "start" : "middle"}>
              {clock(startMs, t)}
            </text>
          </g>
        ))}
        <line x1={LABEL_W} x2={LABEL_W + plotW} y1={LANES.length * LANE_H} y2={LANES.length * LANE_H} stroke="var(--axis)" strokeWidth={1} />
        {LANES.map((lane, li) => {
          const evs = events.filter((e) => lane.types.includes(e.type));
          const y = li * LANE_H + (LANE_H - BAR_H) / 2;
          return (
            <g key={lane.key}>
              <text className="lane-label" x={0} y={li * LANE_H + LANE_H / 2 + 4}>
                {lane.label} <tspan className="muted" style={{ fill: "var(--muted)" }}>({evs.length})</tspan>
              </text>
              {evs.map((e, i) => {
                const x0 = x(e.start_offset_s);
                const w = Math.max(3, x(e.end_offset_s) - x0);
                const hitW = Math.max(w, MIN_HIT);
                const label = `${TYPE_LABEL[e.type] ?? e.type} at ${clock(startMs, e.start_offset_s, true)} UTC, ${e.duration_s} seconds`;
                return (
                  <g
                    key={e.id ?? `${lane.key}-${i}`}
                    className="mark"
                    tabIndex={0}
                    role="button"
                    aria-label={label}
                    onMouseEnter={() => show(e, li)}
                    onMouseLeave={() => setTip(null)}
                    onFocus={() => show(e, li)}
                    onBlur={() => setTip(null)}
                  >
                    <rect x={x0 + w / 2 - hitW / 2} y={li * LANE_H} width={hitW} height={LANE_H} fill="transparent" />
                    <rect className="vis" x={x0} y={y} width={w} height={BAR_H} rx={4} fill={lane.color} />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      {tip && (
        <div className="tip" style={{ left: tip.x, top: tip.y }}>
          <b>{TYPE_LABEL[tip.e.type] ?? tip.e.type}</b> · {clock(startMs, tip.e.start_offset_s, true)}–
          {clock(startMs, tip.e.end_offset_s, true)} UTC
          <br />
          {tip.e.duration_s} s
          {tip.e.confidence != null && ` · confidence ${tip.e.confidence.toFixed(2)}`}
          {tip.e.confidence_basis && ` (${tip.e.confidence_basis.replace("_", "-")})`}
          {tip.e.evidence?.terminated_by && <><br />ended with a {tip.e.evidence.terminated_by}</>}
          {tip.e.reason && <><br />{tip.e.reason}</>}
        </div>
      )}
    </div>
  );
}
