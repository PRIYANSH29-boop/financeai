"use client";

/*
  The design system documenting itself — every token and every component on one page.

  It exists so a change to tokens.css has one obvious place to be reviewed, and so WP2,
  WP3 and WP4 can point at a rendered example instead of re-deriving the rules.

  IMPORTANT: this page contains no product data. The one chart below is drawn from a
  hard-coded synthetic shape and is labelled as such — the "no number is typed into the
  frontend" rule is about the product surfaces, and the way to keep that rule legible is
  for the demo never to look like a result.
*/

import {
  Wordmark, Chip, Badge, Tooltip, Expander, Num,
  CATEGORICAL, CATEGORICAL_ALL_PAIRS_CAP, ACCENT_RAMP, LOSS, CASH, BENCHMARK,
  NUM_NEUTRAL, CHROME, niceTicks, magnitudeColor, seriesColor, needsLegend,
} from "./index";
import { useState } from "react";

const SURFACE_TOKENS = [
  ["--ra-bg", "page plane"],
  ["--ra-surface", "cards, chart surface"],
  ["--ra-surface-2", "insets, tiles"],
  ["--ra-surface-3", "hover, pressed"],
  ["--ra-border", "hairline"],
  ["--ra-border-strong", "input edges"],
];

const INK_TOKENS = [
  ["--ra-text", "15.1:1 on surface"],
  ["--ra-text-muted", "7.3:1"],
  ["--ra-text-faint", "5.3:1 on surface-2"],
  ["--ra-num-neutral", "deltas — information"],
];

const SEMANTIC_TOKENS = [
  ["--ra-accent", "the signature hue"],
  ["--ra-accent-strong", "hover, emphasis"],
  ["--ra-accent-dim", "chart slot 1"],
  ["--ra-loss", "losses + warnings ONLY"],
  ["--ra-cash", "the cash slice, 3.9:1"],
  ["--ra-benchmark", "the “not us” series"],
];

const TYPE_STEPS = [
  ["hero", "3.2rem", "the one figure a page leads with"],
  ["2xl", "2.4rem", "hero supporting"],
  ["xl", "1.8rem", "panel headline number"],
  ["lg", "1.4rem", "section heading"],
  ["md", "1.12rem", "card heading"],
  ["base", "1rem", "body"],
  ["sm", "0.85rem", "secondary body"],
  ["xs", "0.75rem", "captions"],
  ["2xs", "0.68rem", "eyebrow, badge"],
];

const SPACE_STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9];

function Swatch({ token, note }) {
  return (
    <div className="ra-demo-swatch">
      <div className="ra-demo-swatch-chip" style={{ background: `var(${token})` }} />
      <div className="ra-demo-swatch-meta">
        <span className="ra-num">{token.replace("--ra-", "")}</span>
        <span className="ra-faint">{note}</span>
      </div>
    </div>
  );
}

/* A shape, not a result: a fixed synthetic path drawn only to show the chrome. */
const DEMO_A = [8, 14, 11, 22, 30, 26, 38, 47, 43, 58, 66, 61, 74];
const DEMO_B = [8, 12, 13, 17, 21, 24, 27, 31, 34, 38, 41, 45, 48];

function DemoChart() {
  const W = 620, H = 200;
  const { l, r, t, b } = { l: CHROME.pad.l, r: CHROME.pad.r, t: CHROME.pad.t, b: CHROME.pad.b };
  const iw = W - l - r, ih = H - t - b;
  const ticks = niceTicks(0, 80, 5);
  const hi = ticks[ticks.length - 1];
  const x = (i) => l + (i / (DEMO_A.length - 1)) * iw;
  const y = (v) => t + ih - (v / hi) * ih;
  const path = (s) => s.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <div className="ra-card">
      <div className="ra-demo-row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <strong>Chart chrome</strong>
        <Badge variant="loss">synthetic demo data — not a result</Badge>
      </div>
      <svg className="ra-chart" viewBox={`0 0 ${W} ${H}`} height={H} role="img"
           aria-label="Demonstration of the shared chart chrome: gridlines, axis ticks, a series stroke and a dashed benchmark">
        {ticks.map((v) => (
          <g key={v}>
            <line className="ra-chart-grid" x1={l} x2={W - r} y1={y(v)} y2={y(v)} />
            <text className="ra-chart-tick" x={l - 8} y={y(v) + 3.5} textAnchor="end">{v}</text>
          </g>
        ))}
        <line className="ra-chart-axis" x1={l} x2={W - r} y1={y(0)} y2={y(0)} />
        <path className="ra-chart-benchmark" d={path(DEMO_B)} />
        <path className="ra-chart-series" d={path(DEMO_A)} stroke={seriesColor(0)} />
      </svg>
      {needsLegend(2) && (
        <div className="ra-legend" style={{ marginTop: 10 }}>
          <span className="ra-legend-item" style={{ color: seriesColor(0) }}>
            <span className="ra-legend-swatch ra-legend-swatch-line" />
            <span className="ra-muted">Series — slot 1</span>
          </span>
          <span className="ra-legend-item">
            <span className="ra-legend-swatch ra-legend-swatch-dash" />
            <span>Benchmark — muted ink and a dash, never a hue</span>
          </span>
        </div>
      )}
    </div>
  );
}

export default function Gallery() {
  const [preset, setPreset] = useState("balanced");
  const [picks, setPicks] = useState(["AAPL", "MSFT", "JNJ"]);

  return (
    <div className="ra-root ra-page">
      <div className="ra-demo">
        <div className="ra-demo-row" style={{ justifyContent: "space-between" }}>
          <Wordmark size={26} style={{ fontSize: "1.3rem" }} />
          <Badge>WP1 · design system</Badge>
        </div>
        <p className="ra-demo-note" style={{ marginTop: 16 }}>
          Every token and component in the v3 kit, on one page. Colours here were generated
          in OKLCH and measured against this surface, not chosen by eye: ink/surface pairs
          are asserted in <span className="ra-num">tokens.test.js</span>, and both chart
          ramps cleared the data-viz validator&apos;s lightness-band, chroma, colour-vision
          and contrast gates.
        </p>

        {/* ---------------------------------------------------------------- colour */}
        <h2>Colour</h2>
        <h3>Surfaces</h3>
        <div className="ra-demo-grid">
          {SURFACE_TOKENS.map(([t, n]) => <Swatch key={t} token={t} note={n} />)}
        </div>
        <h3>Ink</h3>
        <div className="ra-demo-grid">
          {INK_TOKENS.map(([t, n]) => <Swatch key={t} token={t} note={n} />)}
        </div>
        <h3>Semantics — one accent, and red reserved</h3>
        <div className="ra-demo-grid">
          {SEMANTIC_TOKENS.map(([t, n]) => <Swatch key={t} token={t} note={n} />)}
        </div>
        <p className="ra-demo-note" style={{ marginTop: 12 }}>
          Red is a semantic, not a palette entry: losses and warnings only. It is absent
          from the categorical chart slots below so a series can never be mistaken for a
          loss.
        </p>

        {/* ---------------------------------------------------------------- type */}
        <h2>Type</h2>
        <div className="ra-demo-scale">
          {TYPE_STEPS.map(([k, v, use]) => (
            <div key={k} style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
              <span className="ra-num ra-faint" style={{ width: 96, flex: "none", fontSize: "var(--ra-text-xs)" }}>
                {k} · {v}
              </span>
              <span style={{ fontSize: `var(--ra-text-${k})`, lineHeight: 1.2 }}>
                RankAlpha <span className="ra-num">1,234.56</span>
              </span>
              <span className="ra-faint" style={{ fontSize: "var(--ra-text-xs)", marginLeft: "auto" }}>{use}</span>
            </div>
          ))}
        </div>
        <p className="ra-demo-note" style={{ marginTop: 14 }}>
          Numbers are monospace and tabular everywhere — columns line up and digits do not
          jitter while a slider moves. Tones:{" "}
          <Num>0.00</Num> default · <Num tone="pos">+12.4%</Num> gain ·{" "}
          <Num tone="loss">−18.2%</Num> loss · <Num tone="neutral">+1.0pp</Num> a delta
          that is information, not a loss.
        </p>

        {/* ---------------------------------------------------------------- space */}
        <h2>Space &amp; shape</h2>
        <div className="ra-demo-scale">
          {SPACE_STEPS.map((s) => (
            <div key={s} style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <span className="ra-num ra-faint" style={{ width: 96, flex: "none", fontSize: "var(--ra-text-xs)" }}>
                space-{s}
              </span>
              <span className="ra-demo-space" style={{ width: `var(--ra-space-${s})` }} />
            </div>
          ))}
        </div>
        <div className="ra-demo-row" style={{ marginTop: 20 }}>
          {["sm", "", "lg", "pill"].map((r) => (
            <div key={r || "base"} className="ra-card" style={{
              borderRadius: `var(--ra-radius${r ? `-${r}` : ""})`,
              padding: "14px 18px", fontSize: "var(--ra-text-xs)",
            }}>
              radius{r ? `-${r}` : ""}
            </div>
          ))}
          {[1, 2, 3].map((s) => (
            <div key={s} className="ra-card" style={{
              boxShadow: `var(--ra-shadow-${s})`, padding: "14px 18px", fontSize: "var(--ra-text-xs)",
            }}>
              shadow-{s}
            </div>
          ))}
        </div>

        {/* ---------------------------------------------------------------- components */}
        <h2>Components</h2>

        <h3>Chip — interactive</h3>
        <div className="ra-demo-row">
          {["defensive", "balanced", "aggressive"].map((k) => (
            <Chip key={k} selected={preset === k} onClick={() => setPreset(k)}>
              {k[0].toUpperCase() + k.slice(1)}
            </Chip>
          ))}
          <Chip disabled>Unreachable</Chip>
          <Chip variant="loss" size="sm">stat flagged</Chip>
        </div>
        <div className="ra-demo-row" style={{ marginTop: 12 }}>
          {picks.map((p) => (
            <Chip key={p} selected onDismiss={() => setPicks(picks.filter((x) => x !== p))}
                  dismissLabel={`Remove ${p}`}>
              <span className="ra-num">{p}</span>
            </Chip>
          ))}
          {picks.length === 0 && <span className="ra-faint">All dismissed — reload to reset.</span>}
        </div>

        <h3>Badge — inert</h3>
        <div className="ra-demo-row">
          <Badge>Educational simulation</Badge>
          <Badge variant="accent">Frozen model</Badge>
          <Badge variant="loss">Partial final month</Badge>
          <Badge num>β 1.04</Badge>
          <Badge num variant="accent">Rank IC 0.0505</Badge>
        </div>

        <h3>Tooltip — hover, focus and tap</h3>
        <div className="ra-demo-row">
          <Tooltip text="Return per unit of risk. Higher is better, but it says nothing about how the fall felt.">
            Sharpe
          </Tooltip>
          <Tooltip align="start" text="How much the pie moves when the market moves. Beta 0.5 is roughly half the market's swing." >
            Beta
          </Tooltip>
          <span>
            Cash sleeve{" "}
            <Tooltip text="Cash is the control, not a leftover: it has a beta of zero, so holding more of it pulls the whole pie's beta down." />
          </span>
        </div>

        <h3>Expander</h3>
        <div style={{ display: "grid", gap: 12 }}>
          <Expander summary="How honest is this?" hint="4 things to know" defaultOpen>
            <p className="ra-demo-note" style={{ marginTop: 0 }}>
              The disclosure primitive. Built on <span className="ra-num">&lt;details&gt;</span>,
              so it opens without JavaScript and the browser can find text inside it.
            </p>
          </Expander>
          <Expander summary="What happens after you invest?">
            <p className="ra-demo-note" style={{ marginTop: 0 }}>
              Collapsed by default. The hint line lets a closed expander still say what it
              is hiding.
            </p>
          </Expander>
        </div>

        {/* ---------------------------------------------------------------- charts */}
        <h2>Chart theme</h2>

        <h3>
          Categorical — identity. Fixed order, never cycled, {CATEGORICAL.length} slots
        </h3>
        <div className="ra-demo-grid">
          {CATEGORICAL.map((c) => (
            <div key={c.slot} className="ra-demo-swatch">
              <div className="ra-demo-swatch-chip" style={{ background: c.hex }} />
              <div className="ra-demo-swatch-meta">
                <span className="ra-num">slot {c.slot} · {c.hue}</span>
                <span className="ra-faint ra-num">{c.hex}</span>
              </div>
            </div>
          ))}
        </div>
        <p className="ra-demo-note" style={{ marginTop: 12 }}>
          Seven slots, not eight: red is reserved for losses. Worst adjacent pair clears
          the colour-vision gate at ΔE 13.2 (deutan) and the normal-vision floor at 19.3.
          Where any two marks can sit side by side — scatter, bubble, small multiples —
          only the first <span className="ra-num">{CATEGORICAL_ALL_PAIRS_CAP}</span> slots
          hold; past that, fold the tail into “Other” or facet. A ninth series is never a
          generated hue.
        </p>

        <h3>Magnitude — one hue, {ACCENT_RAMP.length} steps, brightest first</h3>
        <div style={{ display: "flex", borderRadius: "var(--ra-radius-sm)", overflow: "hidden" }}>
          {ACCENT_RAMP.map((hex, i) => (
            <div key={hex} style={{ background: hex, height: 48, flex: 1 }}
                 title={`step ${i + 1} — ${hex}`} />
          ))}
        </div>
        <p className="ra-demo-note" style={{ marginTop: 12 }}>
          What a weight-ordered mark uses. Portfolio weight is a magnitude, not an
          identity — the biggest holding should look biggest, and a twenty-holding pie has
          no honest categorical answer. Twelve marks read off the same ramp:
        </p>
        <div style={{ display: "flex", gap: 2, marginTop: 8 }}>
          {Array.from({ length: 12 }, (_, i) => (
            <div key={i} style={{ background: magnitudeColor(i, 12), height: 34, flex: 1 }} />
          ))}
          <div style={{ background: CASH, height: 34, flex: 1 }} title="cash — reserved" />
        </div>
        <div className="ra-legend" style={{ marginTop: 8 }}>
          <span className="ra-legend-item">
            <span className="ra-legend-swatch" style={{ background: CASH }} />Cash — reserved colour, first-class slice
          </span>
          <span className="ra-legend-item">
            <span className="ra-legend-swatch" style={{ background: LOSS }} />Loss — reserved, never a series
          </span>
          <span className="ra-legend-item">
            <span className="ra-legend-swatch" style={{ background: NUM_NEUTRAL }} />Delta — information, not a loss
          </span>
          <span className="ra-legend-item">
            <span className="ra-legend-swatch" style={{ background: BENCHMARK }} />Benchmark
          </span>
        </div>

        <h3>Chrome</h3>
        <DemoChart />

        {/* ---------------------------------------------------------------- motion */}
        <h2>Motion</h2>
        <p className="ra-demo-note">
          Purposeful only: a panel arriving, a number settling, a slice morphing. Nothing
          loops, nothing moves without a cause, and{" "}
          <span className="ra-num">prefers-reduced-motion</span> removes all of it — the
          layout and every figure are identical either way.
        </p>
        <div className="ra-demo-row" style={{ marginTop: 12 }}>
          <div className="ra-card ra-rise" style={{ fontSize: "var(--ra-text-xs)" }}>
            ra-rise · <span className="ra-num">dur-slow</span> · ease-out
          </div>
          <div className="ra-card ra-fade" style={{ fontSize: "var(--ra-text-xs)" }}>
            ra-fade · <span className="ra-num">dur</span> · ease
          </div>
        </div>

        <p className="ra-demo-note" style={{ marginTop: 40 }}>
          Educational simulation. No real money. Not investment advice.
        </p>
      </div>
    </div>
  );
}
