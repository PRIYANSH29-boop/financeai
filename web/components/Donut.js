"use client";

/*
  Zone 2 — the pie itself. Rebuilt for #32 WP2.

  Cash is a FIRST-CLASS slice, not a remainder rendered as empty space. That is the whole
  honesty argument of the beta engine: a low-beta pie is mostly cash, and a chart that
  hides that is selling a different product from the one being built.

  Three things changed in WP2, each for a stated reason:

  1. COLOUR IS MAGNITUDE, NOT IDENTITY. v1 cycled a 20-entry categorical palette, which
     breaks the one rule categorical colour has — a cycled hue gives two holdings the same
     identity. Portfolio weight is a *magnitude*, so it now reads off the single-hue accent
     ramp in components/ui/chartTheme.js: biggest holding brightest, ordered. Cash keeps
     its own reserved colour and sits outside the ramp; red is never used here at all.
     (Colour follows rank on purpose — that is what a magnitude encoding means. The
     "colour follows the entity, not its rank" rule governs categorical identity, which is
     precisely the encoding this replaced.)

  2. IT MORPHS. Moving the slider used to swap one static chart for another, so the eye had
     to re-read the whole pie. Now the arcs tween between states, which makes "more cash,
     fewer names" legible as a movement rather than a redraw. prefers-reduced-motion cuts
     straight to the final frame — the numbers are identical either way.

  3. THE LEGEND IS THE LIST. The on-slice label used to repeat ticker AND weight, which the
     legend then printed again. Slices now carry the ticker only; the legend carries the
     weight and the cash amount, and is the single place a holding's numbers appear.

  Every figure still comes from the bundle. The tween interpolates ARC GEOMETRY only —
  never a displayed number — so no frame of the animation can show a weight the engine did
  not produce.
*/

import { useEffect, useRef, useState } from "react";
import { money, pct } from "../lib/format";
import { magnitudeColor, CASH } from "./ui/chartTheme";

const SIZE = 360;
const R_OUT = 150;
const R_IN = 96;
const LIFT = 7;               // px a hovered slice steps outward
const CX = SIZE / 2;
const CY = SIZE / 2;
const MORPH_MS = 420;

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

/** WCAG relative luminance — decides whether a slice's label is dark or light ink. */
function isLight(hex) {
  const c = [1, 3, 5].map((i) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 0.4;
}

/**
 * Tween the slice set toward `target`.
 *
 * Colours are carried through from the target rather than recomputed per frame: ranks
 * cross mid-tween, and recomputing would make the ramp flicker. A slice that is arriving
 * grows from zero; one that is leaving shrinks to zero and is then dropped.
 */
function useMorph(target, enabled) {
  const [frame, setFrame] = useState(target);
  const fromRef = useRef(target);
  const rafRef = useRef(0);
  // Identity of the target state — the tween restarts only when the pie actually changes.
  const sig = target.map((s) => `${s.key}:${s.weight.toFixed(6)}`).join("|");

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    if (!enabled || reduce) {
      fromRef.current = target;
      setFrame(target);
      return undefined;
    }

    const byKey = new Map();
    target.forEach((s) => byKey.set(s.key, { ...s, from: 0, to: s.weight }));
    fromRef.current.forEach((s) => {
      const hit = byKey.get(s.key);
      if (hit) hit.from = s.weight;
      // A departing slice keeps its OLD colour on the way out; borrowing a ramp step it
      // no longer owns would recolour the survivors around it.
      else byKey.set(s.key, { ...s, from: s.weight, to: 0 });
    });
    const lanes = [...byKey.values()];

    const t0 = performance.now();
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / MORPH_MS);
      const e = 1 - (1 - p) ** 3;                       // ease-out cubic
      setFrame(
        lanes
          .map((l) => ({ ...l, weight: l.from + (l.to - l.from) * e }))
          .filter((l) => l.weight > 1e-6),
      );
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
      else { fromRef.current = target; setFrame(target); }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // `target` is captured through `sig`; depending on the array itself would restart the
    // tween on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, enabled]);

  return frame;
}

export default function Donut({ holdings, cashWeight, capital, currency, selected, onSelect }) {
  const [hovered, setHovered] = useState(null);

  // Sorted big-to-small so the magnitude ramp actually encodes magnitude, and so the eye
  // reads the pie in the order the weights matter.
  const ranked = [...holdings].sort((a, b) => b.weight - a.weight);
  const target = [
    ...ranked.map((h, i) => ({
      key: h.ticker,
      label: h.ticker,
      name: h.name,
      weight: h.weight,
      color: magnitudeColor(i, ranked.length),
    })),
    ...(cashWeight > 0.0005
      ? [{ key: "CASH", label: "Cash", name: "Cash sleeve", weight: cashWeight, color: CASH }]
      : []),
  ];

  const frame = useMorph(target, true);

  const total = frame.reduce((s, x) => s + x.weight, 0) || 1;
  let angle = -Math.PI / 2;           // start at 12 o'clock
  const LABEL_MIN = 0.05;             // only labels that comfortably fit go on the pie
  const rMid = (R_OUT + R_IN) / 2;

  const arcs = frame.map((s) => {
    const sweep = (s.weight / total) * Math.PI * 2;
    const a0 = angle;
    angle += sweep;
    const mid = a0 + sweep / 2;
    const lift = hovered === s.key ? LIFT : 0;
    const dx = Math.cos(mid) * lift;
    const dy = Math.sin(mid) * lift;
    return {
      ...s,
      mid,
      dx, dy,
      fitsLabel: s.weight >= LABEL_MIN,
      lx: CX + rMid * Math.cos(mid) + dx,
      ly: CY + rMid * Math.sin(mid) + dy,
      hx: CX + (R_OUT + 16) * Math.cos(mid) + dx,
      hy: CY + (R_OUT + 16) * Math.sin(mid) + dy,
      d: arc(CX, CY, R_OUT, R_IN, a0, Math.max(angle - 0.004, a0 + 0.0005)),
    };
  });

  const invested = 1 - cashWeight;
  const hoveredArc = arcs.find((a) => a.key === hovered);

  return (
    <div className="donut-wrap">
      <svg className="donut" viewBox={`-14 -14 ${SIZE + 28} ${SIZE + 28}`}
           width="100%" style={{ maxWidth: SIZE }}
           role="img" aria-label="Portfolio allocation by holding, including cash">
        {arcs.map((a) => (
          <path
            key={a.key}
            d={a.d}
            fill={a.color}
            transform={`translate(${a.dx.toFixed(2)} ${a.dy.toFixed(2)})`}
            className={`slice${selected && selected !== a.key ? " dim" : ""}`}
            onClick={() => onSelect(a.key)}
            onMouseEnter={() => setHovered(a.key)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(a.key)}
            onBlur={() => setHovered(null)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(a.key)}
            tabIndex={0}
            role="button"
            aria-label={`${a.label}, ${pct(a.weight)} of the pie. Show why it is held.`}
          >
            <title>{`${a.label} — ${pct(a.weight)} · ${money(a.weight * capital, currency)}`}</title>
          </path>
        ))}

        {/* Ticker only. The weight lives in the legend below, said once. */}
        {arcs.filter((a) => a.fitsLabel).map((a) => (
          <text key={`lbl-${a.key}`} className="slice-label" x={a.lx} y={a.ly}
                textAnchor="middle" dominantBaseline="middle"
                fill={isLight(a.color) ? "var(--bg)" : "var(--text)"}
                opacity={selected && selected !== a.key ? 0.25 : 1}>
            {a.label}
          </text>
        ))}

        {/* The click affordance: on hover the slice steps out and says what clicking does.
            Without it the pie looked like a picture rather than a control. */}
        {hoveredArc && (
          <g className="why-hint" pointerEvents="none">
            <rect x={hoveredArc.hx - 20} y={hoveredArc.hy - 10} width="40" height="20"
                  rx="10" fill="var(--surface-3)" stroke="var(--border-strong)" />
            <text x={hoveredArc.hx} y={hoveredArc.hy + 1} textAnchor="middle"
                  dominantBaseline="middle" fontSize="10" fill="var(--text)">
              why?
            </text>
          </g>
        )}

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

      {/* The legend IS the holdings list — every slice named once, with its weight and its
          share of the user's capital. Thin slices that lost their on-pie label are still
          fully described here. */}
      <ul className="pie-legend">
        {target.map((a) => (
          <li key={`leg-${a.key}`}
              className={selected && selected !== a.key ? "dim" : undefined}
              onClick={() => onSelect(a.key)} tabIndex={0} role="button"
              onMouseEnter={() => setHovered(a.key)} onMouseLeave={() => setHovered(null)}
              aria-label={`${a.label}, ${pct(a.weight, 1)}, ${money(a.weight * capital, currency)}`}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(a.key)}>
            <span className="sw" style={{ background: a.color }} />
            <span className="leg-tk">{a.label}</span>
            <span className="leg-wt num">{pct(a.weight, 1)}</span>
            <span className="leg-amt num">{money(a.weight * capital, currency)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
