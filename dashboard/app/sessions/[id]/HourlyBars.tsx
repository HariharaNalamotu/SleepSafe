"use client";

import { useEffect, useRef, useState } from "react";

const LEFT = 36;
const RIGHT = 8;
const TOP = 12;
const PLOT_H = 150;
const AXIS_H = 24;

/** Events per hour of sleep for each clock hour: one series, one color, top-rounded bars on a zero baseline. */
export default function HourlyBars({ values, startUtc }: { values: (number | null)[]; startUtc: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(700);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(240, Math.floor(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const max = Math.max(5, ...values.map((v) => v ?? 0));
  const top = Math.ceil(max / 5) * 5;
  const ticks = Array.from({ length: 5 }, (_, i) => (top / 4) * i);
  const plotW = width - LEFT - RIGHT;
  const slot = plotW / values.length;
  const barW = Math.max(6, Math.min(56, slot - 2 - slot * 0.35));
  const y = (v: number) => TOP + PLOT_H - (v / top) * PLOT_H;
  const startMs = new Date(startUtc).getTime();
  const hourLabel = (i: number) => new Date(startMs + i * 3600_000).toISOString().slice(11, 16);

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={TOP + PLOT_H + AXIS_H} role="img" aria-label="Breathing events per hour of sleep, by hour">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={LEFT} x2={LEFT + plotW} y1={y(t)} y2={y(t)} stroke={t === 0 ? "var(--axis)" : "var(--grid)"} strokeWidth={1} />
            <text x={LEFT - 6} y={y(t) + 4} textAnchor="end">{Math.round(t)}</text>
          </g>
        ))}
        {values.map((v, i) => {
          const cx = LEFT + slot * i + slot / 2;
          const x0 = cx - barW / 2;
          const h = v ? (v / top) * PLOT_H : 0;
          const r = Math.min(4, h);
          const base = y(0);
          return (
            <g key={i} tabIndex={0} role="button"
               aria-label={`${hourLabel(i)} UTC: ${v == null ? "no sleep" : `${v} events per hour`}`}
               onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
               onFocus={() => setHover(i)} onBlur={() => setHover(null)}>
              <rect x={LEFT + slot * i} y={TOP} width={slot} height={PLOT_H} fill="transparent" />
              {h > 0 && (
                <path
                  d={`M${x0},${base} V${base - h + r} Q${x0},${base - h} ${x0 + r},${base - h} H${x0 + barW - r} Q${x0 + barW},${base - h} ${x0 + barW},${base - h + r} V${base} Z`}
                  fill="var(--series-1)" opacity={hover == null || hover === i ? 1 : 0.55}
                />
              )}
              <text x={cx} y={TOP + PLOT_H + 17} textAnchor="middle">{hourLabel(i)}</text>
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <div className="tip" style={{ left: LEFT + slot * hover + slot / 2, top: y(values[hover] ?? 0) }}>
          <b>{hourLabel(hover)} UTC</b>
          <br />
          {values[hover] == null ? "no estimated sleep" : `${values[hover]} events / hour of sleep`}
        </div>
      )}
    </div>
  );
}
