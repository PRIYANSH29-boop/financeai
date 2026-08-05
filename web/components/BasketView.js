"use client";

/*
  Module B — Build your basket (Phase 23). "If I pick these stocks, what would that have
  been?" answered strictly in the PAST TENSE. All math is client-side, from the per-stock
  monthly series in stocks.json, via lib/basket.js (the node:test-verified reference).

  HARD RULE: the ONLY answer to the horizon question is the historical rolling-12-month
  distribution RANGE. There is no expected-return number anywhere on this page, by design.
*/

import { useMemo, useState } from "react";
import { pct, num, beta as fmtBeta } from "../lib/format";
import { basketScorecard, outcomeDistribution } from "../lib/basket";
import { partialMonthNote } from "../lib/disclosure";

const MAX_PICKS = 10;

function Stat({ k, v, tone }) {
  return (
    <div className="tile">
      <div className="k">{k}</div>
      <div className={`v ${tone || ""}`}>{v}</div>
    </div>
  );
}

// Nearest preset pie to the basket's realised beta — the "you vs the machine" comparison.
function matchedPreset(index, basketBeta) {
  if (basketBeta == null || Number.isNaN(basketBeta)) return null;
  let best = null;
  for (const p of index.presets) {
    const d = Math.abs(p.beta - basketBeta);
    if (!best || d < best.d) best = { ...p, d };
  }
  return best;
}

export default function BasketView({ stocks, index, explore }) {
  const [picks, setPicks] = useState([]);
  const [q, setQ] = useState("");

  // Universe of pickable (scored) names, with display names.
  const pickable = useMemo(() => {
    const stat = stocks.stats || {};
    return Object.keys(stocks.returns || {})
      .map((tk) => ({ ticker: tk, name: stat[tk]?.name || tk }))
      .sort((a, b) => a.ticker.localeCompare(b.ticker));
  }, [stocks]);

  const matches = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return [];
    return pickable
      .filter((s) => !picks.includes(s.ticker))
      .filter((s) => `${s.ticker} ${s.name}`.toLowerCase().includes(needle))
      .slice(0, 8);
  }, [q, pickable, picks]);

  const add = (tk) => {
    if (picks.length >= MAX_PICKS || picks.includes(tk)) return;
    setPicks((p) => [...p, tk]);
    setQ("");
  };
  const remove = (tk) => setPicks((p) => p.filter((x) => x !== tk));

  const result = useMemo(() => {
    if (picks.length === 0) return null;
    const sc = basketScorecard(picks, stocks);
    const dist = outcomeDistribution(picks, stocks, 12);
    return { sc, dist };
  }, [picks, stocks]);

  const preset = result ? matchedPreset(index, result.sc.beta) : null;

  return (
    <main className="wrap">
      <div className="stack" style={{ paddingTop: 22 }}>
        <div className="card">
          <h2 style={{ marginTop: 0 }}>Build your basket</h2>
          <p className="muted" style={{ marginTop: 4 }}>
            Pick up to {MAX_PICKS} stocks and see what that equal-weight basket{" "}
            <strong>would have done</strong> over the committed history — measured, past
            tense, never a forecast. Only names the frozen model scores (the S&amp;P 500) can
            be picked: <span className="faint">the model can&apos;t speak for companies it
            never trained on.</span>
          </p>

          <div className="basket-picker">
            <input className="explore-search" type="search" value={q}
                   placeholder={picks.length >= MAX_PICKS
                     ? `Basket full (${MAX_PICKS} max)` : "Add a stock — ticker or name…"}
                   disabled={picks.length >= MAX_PICKS}
                   onChange={(e) => setQ(e.target.value)} />
            {matches.length > 0 && (
              <ul className="picker-menu">
                {matches.map((s) => (
                  <li key={s.ticker} onClick={() => add(s.ticker)}>
                    <span className="num">{s.ticker}</span>
                    <span className="muted">{s.name}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="basket-chips">
            {picks.length === 0 && (
              <span className="empty-state">No stocks yet — add a few to begin.</span>
            )}
            {picks.map((tk) => (
              <span key={tk} className="pick-chip">
                {tk}
                <button aria-label={`Remove ${tk}`} onClick={() => remove(tk)}>×</button>
              </span>
            ))}
            {picks.length > 0 && (
              <span className="faint" style={{ fontSize: "0.8rem", alignSelf: "center" }}>
                equal-weight · {picks.length}/{MAX_PICKS}
              </span>
            )}
          </div>
        </div>

        {result && result.sc.n_months > 0 && (
          <>
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Basket scorecard — historical, equal-weight</h3>
              <div className="tiles">
                <Stat k="Realised beta" v={fmtBeta(result.sc.beta)} />
                <Stat k="Sharpe" v={num(result.sc.sharpe)} />
                <Stat k="Sortino" v={num(result.sc.sortino)} />
                <Stat k="Ann. vol" v={pct(result.sc.ann_vol, 0)} />
                <Stat k="Max drawdown" v={pct(result.sc.max_drawdown, 1)} tone="loss" />
                <Stat k="Hit rate" v={pct(result.sc.hit_rate, 0)} />
              </div>
              <p className="faint" style={{ fontSize: "0.8rem", marginTop: 10 }}>
                Over {result.sc.n_months} months ({result.sc.start} → {result.sc.end}), equal
                weighted, monthly. Survivorship-biased universe. Past behaviour, not a promise.
              </p>
            </div>

            {preset && (
              <div className="card">
                <h3 style={{ marginTop: 0 }}>Your picks vs the machine, at matched beta</h3>
                <p className="muted" style={{ marginTop: 4 }}>
                  Your basket&apos;s realised beta is{" "}
                  <span className="num">β{fmtBeta(result.sc.beta)}</span>. The engine&apos;s
                  closest pie is <strong>{preset.label}</strong>{" "}
                  (<span className="num">β{preset.beta.toFixed(2)}</span>) — a diversified{" "}
                  book capped at 8%/stock, 30%/sector, ≤5 per sector, versus your{" "}
                  {picks.length}-stock equal-weight pick.
                </p>
                <p className="faint" style={{ fontSize: "0.82rem" }}>
                  Open the “Build a pie” tab at {preset.label} to see the machine&apos;s full
                  scorecard side by side. Concentration is the trade-off: fewer names means
                  more idiosyncratic risk than the capped pie carries.
                </p>
              </div>
            )}

            <div className="card">
              <h3 style={{ marginTop: 0 }}>
                If you had held this basket for 12 months…
              </h3>
              {result.dist.n > 0 ? (
                <>
                  <div className="range-row">
                    <div className="range-end loss">
                      <div className="range-k">worst 12-mo</div>
                      <div className="range-v num">{pct(result.dist.min, 1)}</div>
                    </div>
                    <div className="range-end">
                      <div className="range-k">median 12-mo</div>
                      <div className="range-v num">{pct(result.dist.median, 1)}</div>
                    </div>
                    <div className="range-end num-pos">
                      <div className="range-k">best 12-mo</div>
                      <div className="range-v num">{pct(result.dist.max, 1)}</div>
                    </div>
                  </div>
                  <p className="muted" style={{ marginTop: 12, fontSize: "0.86rem" }}>
                    Across every overlapping 12-month window in the history ({result.dist.n}{" "}
                    of them), this basket would have landed between{" "}
                    <span className="num">{pct(result.dist.min, 1)}</span> and{" "}
                    <span className="num">{pct(result.dist.max, 1)}</span>, median{" "}
                    <span className="num">{pct(result.dist.median, 1)}</span>.
                  </p>
                  <p className="faint" style={{ fontSize: "0.82rem", marginTop: 6 }}>
                    This range is the <strong>only</strong> answer to “what will I get?”. It
                    is what happened, not what will happen — no expected return is shown or
                    implied, because none can be honestly promised.
                  </p>
                </>
              ) : (
                <p className="muted">
                  Not enough shared history among these picks for a 12-month window yet — add
                  names with longer track records.
                </p>
              )}
            </div>

            <div className="caveats">
              <strong>Read this before you believe any number above.</strong>
              <ul>{index.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </div>
          </>
        )}

        {result && result.sc.n_months === 0 && (
          <div className="card">
            <p className="muted">These picks share no overlapping monthly history.</p>
          </div>
        )}

        <p className="faint" style={{ fontSize: "0.8rem", margin: 0 }}>
          Basket math from per-stock series as of {stocks.as_of}. Benchmark:{" "}
          {stocks.benchmark_label}.
          {partialMonthNote(stocks) ? <> ⚠ {partialMonthNote(stocks)}</> : null}
        </p>
      </div>
    </main>
  );
}
