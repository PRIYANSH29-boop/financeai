"use client";

/*
  Zone 2 — the pie itself.

  Cash is a FIRST-CLASS slice, not a remainder rendered as empty space. That is the whole
  honesty argument of the beta engine: a low-beta pie is mostly cash, and a chart that
  hides that is selling a different product from the one being built.
*/

import { money, pct } from "../lib/format";

const SIZE = 340;
const R_OUT = 150;
const R_IN = 96;
const CX = SIZE / 2;
const CY = SIZE / 2;

// Neutral-leaning sequential ramp; the accent anchors it and cash sits outside it.
const PALETTE = [
  "#0d8b80", "#1a9b8f", "#2faa9c", "#48b8a8", "#63c5b5", "#7fd1c2",
  "#3d6f9e", "#4e80ad", "#6191bb", "#75a2c8", "#8bb3d4", "#a2c4df",
  "#7a6ba8", "#8b7cb5", "#9d8dc1", "#af9fcd", "#c1b1d8", "#d2c4e3",
  "#a8874f", "#b89a64",
];

function arc(cx, cy, rOut, rIn, a0, a1) {
  const p = (r, a) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const [x0, y0] = p(rOut, a0);
  const [x1, y1] = p(rOut, a1);
  const [x2, y2] = p(rIn, a1);
  const [x3, y3] = p(rIn, a0);
  return `M${x0},${y0} A${rOut},${rOut} 0 ${large} 1 ${x1},${y1} ` +
         `L${x2},${y2} A${rIn},${rIn} 0 ${large} 0 ${x3},${y3} Z`;
}

export default function Donut({ holdings, cashWeight, capital, currency, selected, onSelect }) {
  const slices = [
    ...holdings.map((h, i) => ({
      key: h.ticker, label: h.ticker, weight: h.weight, color: PALETTE[i % PALETTE.length],
    })),
    ...(cashWeight > 0.0005
      ? [{ key: "CASH", label: "Cash", weight: cashWeight, color: "var(--cash)" }]
      : []),
  ];

  const total = slices.reduce((s, x) => s + x.weight, 0) || 1;
  let angle = -Math.PI / 2;           // start at 12 o'clock
  const LABEL_MIN = 0.05;             // only labels that comfortably fit go on the pie
  const rMid = (R_OUT + R_IN) / 2;
  const arcs = slices.map((s) => {
    const sweep = (s.weight / total) * Math.PI * 2;
    const a0 = angle;
    angle += sweep;
    const mid = a0 + sweep / 2;
    return {
      ...s,
      fitsLabel: s.weight >= LABEL_MIN,
      lx: CX + rMid * Math.cos(mid),
      ly: CY + rMid * Math.sin(mid),
      d: arc(CX, CY, R_OUT, R_IN, a0, Math.max(angle - 0.004, a0 + 0.0005)),
    };
  });

  const invested = 1 - cashWeight;

  return (
    <div className="donut-wrap">
      <svg className="donut" viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE}
           role="img" aria-label="Portfolio allocation by holding, including cash">
        {arcs.map((a) => (
          <path
            key={a.key}
            d={a.d}
            fill={a.color}
            className={selected && selected !== a.key ? "dim" : undefined}
            onClick={() => onSelect(a.key)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(a.key)}
            tabIndex={0}
            role="button"
            aria-label={`${a.label}, ${pct(a.weight)} of the pie`}
          >
            <title>{`${a.label} — ${pct(a.weight)} · ${money(a.weight * capital, currency)}`}</title>
          </path>
        ))}

        {/* On-slice labels (ticker + weight%) where the slice is wide enough; thin slices
            fall through to the legend below so nothing overlaps. */}
        {arcs.filter((a) => a.fitsLabel).map((a) => (
          <text key={`lbl-${a.key}`} className="slice-label" x={a.lx} y={a.ly}
                textAnchor="middle" dominantBaseline="middle"
                opacity={selected && selected !== a.key ? 0.25 : 1}>
            <tspan x={a.lx} dy="-0.15em">{a.label}</tspan>
            <tspan x={a.lx} dy="1.05em" className="slice-label-pct">{pct(a.weight, 0)}</tspan>
          </text>
        ))}

        <text className="donut-centre-label" x={CX} y={CY - 18} textAnchor="middle">
          Invested
        </text>
        <text className="donut-centre-value" x={CX} y={CY + 8} textAnchor="middle">
          {money(invested * capital, currency)}
        </text>
        <text className="donut-centre-sub" x={CX} y={CY + 28} textAnchor="middle">
          {holdings.length} stocks · {pct(cashWeight)} cash
        </text>
      </svg>

      {/* Full legend — every slice, so thin ones (below the on-pie label threshold) are
          still named with their exact weight. Click to select, same as the arcs. */}
      <ul className="pie-legend">
        {arcs.map((a) => (
          <li key={`leg-${a.key}`}
              className={selected && selected !== a.key ? "dim" : undefined}
              onClick={() => onSelect(a.key)} tabIndex={0} role="button"
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(a.key)}>
            <span className="sw" style={{ background: a.color }} />
            <span className="leg-tk">{a.label}</span>
            <span className="leg-wt num">{pct(a.weight, 1)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
