"use client";

/*
  Zones 1, 3, 4, 5 and the "How it works" modal.

  Every figure rendered here is read straight off the bundle. Nothing is computed beyond
  multiplying a weight by the user's capital, which is exactly the client-side multiplier
  the bundle is designed for.
*/

import { useState } from "react";
import { money, pct, num, pp, beta as fmtBeta } from "../lib/format";
import { GrowthChart, DrawdownChart } from "./Charts";
import { partialMonthNote } from "../lib/disclosure";

/* ------------------------------------------------------------------ tooltip */
export function Info({ text }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="info"
          onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button type="button" aria-label={text} onFocus={() => setOpen(true)}
              onBlur={() => setOpen(false)}>i</button>
      {open && <span className="bubble" role="tooltip">{text}</span>}
    </span>
  );
}

/* ------------------------------------------------------------------ zone 1 */
export function ControlBar({ index, capital, setCapital, target, setTarget, pie }) {
  const max = index.max_achievable_beta;
  const grid = index.betas;
  const reachable = grid.filter((b) => b <= max);
  const unreachableFrac = 1 - reachable.length / grid.length;

  const snap = (raw) =>
    reachable.reduce((best, b) => (Math.abs(b - raw) < Math.abs(best - raw) ? b : best),
                     reachable[0]);

  return (
    <div className="card">
      <div className="control-grid">
        <div>
          <label className="field" htmlFor="capital">How much would you invest?</label>
          <div className="capital-input">
            <span className="prefix">£</span>
            <input id="capital" type="number" min="1" step="100" value={capital}
                   onChange={(e) => setCapital(Math.max(1, Number(e.target.value) || 0))} />
          </div>
          <div className="faint" style={{ fontSize: "0.78rem", marginTop: 6 }}>
            Simulated. No money moves.
          </div>
        </div>

        <div>
          <label className="field" htmlFor="beta">
            Risk level — target beta{" "}
            <Info text="Beta is how much your pie moves when the market moves. Beta 0.5 means roughly half the market's swing. Cash is how we lower it." />
          </label>
          <input id="beta" type="range" min={grid[0]} max={grid[grid.length - 1]}
                 step={index.beta_step} value={target}
                 onChange={(e) => setTarget(snap(Number(e.target.value)))} />
          <div className="beta-scale">
            {unreachableFrac > 0 && (
              <div className="unreachable" style={{ width: `${unreachableFrac * 100}%` }}
                   title={`Above ${fmtBeta(max)} is not achievable long-only`} />
            )}
          </div>
          <div className="scale-labels">
            <span>{fmtBeta(grid[0])}</span>
            <span className="faint">
              max achievable long-only {fmtBeta(max)}
            </span>
            <span>{fmtBeta(grid[grid.length - 1])}</span>
          </div>
          <div className="chips">
            {index.presets.map((p) => (
              <button key={p.key} className="chip" type="button"
                      aria-pressed={Math.abs(p.beta - target) < 1e-9}
                      onClick={() => setTarget(p.beta)}>
                {p.label} β{p.beta.toFixed(2)}
              </button>
            ))}
          </div>
        </div>

        <div className="readout">
          <span className="muted" style={{ fontSize: "0.82rem" }}>
            Cash sleeve{" "}
            <Info text="Cash is how we lower risk — your shock absorber. It has a beta of zero, so the more cash you hold, the less your pie moves with the market." />
          </span>
          <span className="big">{pct(pie.cash_weight)}</span>
          <span className="num muted">{money(pie.cash_weight * capital, index.currency)}</span>
          <span className="muted" style={{ marginTop: 8, fontSize: "0.86rem" }}>
            Invested{" "}
            <strong className="num">
              {money((1 - pie.cash_weight) * capital, index.currency)}
            </strong>{" "}
            across {pie.n_holdings} stocks
          </span>
          {pie.beta_capped && (
            <span className="loss" style={{ fontSize: "0.8rem", marginTop: 6 }}>
              Target β{target.toFixed(2)} is not reachable long-only — showing the highest
              achievable, β{fmtBeta(pie.achieved_beta)}.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ zone 3 */
export function WhyHolding({ pie, selected, capital, currency, guardrails }) {
  if (!selected) {
    return (
      <div className="card">
        <h2>Why this holding</h2>
        <p className="empty-state">Click any slice to see why it&apos;s in the pie.</p>
      </div>
    );
  }

  if (selected === "CASH") {
    return (
      <div className="card">
        <h2>Why this holding</h2>
        <div className="holding-head" style={{ marginTop: 12 }}>
          <span className="ticker">Cash</span>
          <span className="num" style={{ fontSize: "1.1rem" }}>{pct(pie.cash_weight)}</span>
        </div>
        <div className="muted num" style={{ fontSize: "0.86rem" }}>
          {money(pie.cash_weight * capital, currency)}
        </div>
        <p style={{ marginTop: 14 }}>
          Cash is not left-over — it is the control. Cash has a beta of zero, so holding
          more of it pulls the whole pie&apos;s beta down. To hit your target of{" "}
          <span className="num">β{fmtBeta(pie.target_beta)}</span> the engine holds{" "}
          <span className="num">{pct(pie.cash_weight)}</span> in cash alongside{" "}
          {pie.n_holdings} stocks, giving a realised{" "}
          <span className="num">β{fmtBeta(pie.achieved_beta)}</span>.
        </p>
        <p className="muted" style={{ fontSize: "0.86rem" }}>
          The pie is never levered. If a target beta cannot be reached by holding less cash,
          the engine caps it and says so rather than borrowing to fake it.
        </p>
      </div>
    );
  }

  const h = pie.holdings.find((x) => x.ticker === selected);
  if (!h) return null;
  const capPct = Math.min(1, h.pct_of_cap);

  return (
    <div className="card">
      <h2>Why this holding</h2>
      <div className="holding-head" style={{ marginTop: 12 }}>
        <div>
          <div className="ticker">{h.ticker}</div>
          <div className="muted" style={{ fontSize: "0.9rem" }}>{h.name}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="num" style={{ fontSize: "1.1rem" }}>{pct(h.weight)}</div>
          <div className="muted num" style={{ fontSize: "0.84rem" }}>
            {money(h.weight * capital, currency)}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 10 }}>
        <span className="sector-tag">{h.sector}</span>{" "}
        <span className="sector-tag num">β {fmtBeta(h.beta)}</span>
      </div>

      <ul className="reason-list">
        {h.reasons.map((r, i) => <li key={i}>{r}</li>)}
      </ul>

      <div style={{ marginTop: 16 }}>
        <div className="muted" style={{ fontSize: "0.8rem", marginBottom: 5 }}>
          Weight vs the {pct(guardrails.name_cap, 0)} single-stock cap
        </div>
        <div className="cap-bar"><div style={{ width: `${capPct * 100}%` }} /></div>
        <div className="faint num" style={{ fontSize: "0.76rem", marginTop: 4 }}>
          {pct(h.weight)} of a {pct(guardrails.name_cap, 0)} maximum
          {h.at_cap ? " — at the cap" : ""}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ zone 4 */
export function DriftPanel({ pie }) {
  const drift = pie.drift;
  if (!drift?.available) {
    return (
      <details className="expander">
        <summary>What happens after you invest?</summary>
        <div className="body muted">
          Drift simulation unavailable for this pie ({drift?.reason || "no data"}).
        </div>
      </details>
    );
  }
  const rows = drift.holdings.slice(0, 12);
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.delta_pp)), 0.01);

  return (
    <details className="expander">
      <summary>What happens after you invest?</summary>
      <div className="body">
        <p className="muted" style={{ marginTop: 0 }}>
          You buy the target pie, then prices move. By the next rebalance your weights are
          no longer the ones you chose — winners grow, losers shrink, and cash holds still.
          Below is one rebalance period of real price history applied to today&apos;s
          weights. Rebalancing sells a little of what grew and tops up what shrank, to put
          the pie back on target.
        </p>
        <div className="faint" style={{ fontSize: "0.78rem", marginBottom: 10 }}>
          Period ending {drift.period_end}. Changes are in percentage points of weight —
          not gains or losses.
        </div>
        {rows.map((r) => {
          const frac = Math.abs(r.delta_pp) / maxAbs / 2;
          const left = r.delta_pp >= 0 ? 50 : 50 - frac * 100;
          return (
            <div className="drift-row" key={r.ticker}>
              <span className="num">{r.ticker}</span>
              <span className="drift-track">
                <span className="zero" />
                <span className="bar" style={{ left: `${left}%`, width: `${frac * 100}%` }} />
              </span>
              <span className="num" style={{ textAlign: "right", color: "var(--num-neutral)" }}>
                {pp(r.delta_pp)}
              </span>
            </div>
          );
        })}
        <div className="drift-row" style={{ borderTop: "1px solid var(--border)", marginTop: 8, paddingTop: 8 }}>
          <span className="num">Cash</span>
          <span className="muted" style={{ fontSize: "0.82rem" }}>
            {pct(drift.cash.target_weight)} → {pct(drift.cash.drifted_weight)}
          </span>
          <span className="num" style={{ textAlign: "right", color: "var(--num-neutral)" }}>
            {pp(drift.cash.delta_pp)}
          </span>
        </div>
      </div>
    </details>
  );
}

/* ------------------------------------------------------------------ zone 5 */
export function HonestyPanel({ pie, index, capital }) {
  const sc = pie.scorecard;
  const bd = pie.beta_in_drawdown_window;
  // #21 measured regime beta for this preset, if the bundle carries it (only the 3 presets).
  const regime = index.regime_beta?.[pie.target_beta?.toFixed(2)];

  return (
    <details className="expander" open>
      <summary>How honest is this?</summary>
      <div className="body">
        <h3 style={{ marginBottom: 8 }}>Growth of {money(capital, index.currency)}</h3>
        <GrowthChart series={pie.equity} benchmark={index.benchmark.equity}
                     capital={capital} currency={index.currency} />
        <div className="legend" style={{ marginBottom: 18 }}>
          <span><span className="sw" style={{ background: "var(--accent)" }} />This pie</span>
          <span><span className="sw" style={{ background: "var(--text-faint)" }} />{index.benchmark.label}</span>
        </div>

        <h3 style={{ marginBottom: 8 }}>Drawdown — how far it fell from its peak</h3>
        <DrawdownChart series={pie.drawdown} />

        <div className="tiles" style={{ marginTop: 18 }}>
          <div className="tile">
            <div className="k">Sharpe</div>
            <div className="v">{num(sc.sharpe)}</div>
            <div className="sub">return per unit of risk</div>
          </div>
          <div className="tile">
            <div className="k">Max drawdown</div>
            <div className="v loss">{pct(sc.max_drawdown)}</div>
            <div className="sub">worst peak-to-trough fall</div>
          </div>
          <div className="tile">
            <div className="k">Realised beta</div>
            <div className="v">{fmtBeta(pie.achieved_beta)}</div>
            <div className="sub">
              target {fmtBeta(pie.target_beta)}
              {bd?.beta !== null && bd?.beta !== undefined && (
                <>
                  {" · "}
                  <strong>≈β {fmtBeta(bd.beta)} inside the drawdown window</strong>
                </>
              )}
            </div>
          </div>
        </div>

        {bd?.beta !== null && bd?.beta !== undefined && (
          <p className="muted" style={{ fontSize: "0.84rem", marginTop: 12 }}>
            Between {bd.start} and {bd.end} — the pie&apos;s worst stretch — its realised
            beta was <span className="num">{fmtBeta(bd.beta)}</span>, against a headline{" "}
            <span className="num">{fmtBeta(pie.achieved_beta)}</span>. A beta estimated over
            calm months is not a promise about how it behaves in a fall.
          </p>
        )}

        {regime && regime.calm != null && regime.stressed_core != null && (
          <p className="muted" style={{ fontSize: "0.84rem", marginTop: 12 }}>
            <strong>Measured beta drift (2022 bear):</strong> across the committed history
            this pie&apos;s realised beta was about{" "}
            <span className="num">β{fmtBeta(regime.calm)}</span> in calm months but about{" "}
            <span className="num">β{fmtBeta(regime.stressed_core)}</span> in the{" "}
            {regime.n_core}-month drawdown core of the 2022 bear — the target-beta estimate
            understates risk exactly when it matters.{" "}
            <span className="faint">
              Single stress episode, in-sample characterisation — directional, not a forecast.
            </span>
          </p>
        )}

        {partialMonthNote(index) ? (
          <p className="muted" style={{ fontSize: "0.84rem", marginTop: 12 }}>
            ⚠ {partialMonthNote(index)}
          </p>
        ) : null}

        <div className="caveats" style={{ marginTop: 16 }}>
          <strong>Read this before you believe any number above.</strong>
          <ul>{index.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </div>
      </div>
    </details>
  );
}

/* ------------------------------------------------------------------ modal */
export function HowItWorks({ steps, onClose }) {
  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true"
         aria-label="How it works">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>How it works</h2>
        <ol>{steps.map((s, i) => <li key={i}>{s}</li>)}</ol>
        <p className="muted" style={{ fontSize: "0.86rem" }}>
          The model is frozen — it was trained once and is never refitted to make these
          numbers look better.
        </p>
        <div style={{ display: "flex", gap: 14, alignItems: "center", marginTop: 16 }}>
          <a href="https://github.com/PRIYANSH29-boop/financeai" target="_blank"
             rel="noreferrer">See the code and the research →</a>
          <button className="chip" style={{ marginLeft: "auto" }} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
