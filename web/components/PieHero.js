"use client";

/*
  The hero — #32 WP2.

  The cold read of v1 was that the page "leads with an input instead of a result": the
  first thing on screen was a capital box and a slider, and you had to work out what the
  product even produced. This band inverts that. It states the RESULT for the default
  preset — a built pie, its size, its risk, and how it behaved — before asking for
  anything. The controls still sit directly beneath it, so nothing is hidden; they are
  just no longer the opening move.

  Every figure is read straight off the bundle. The only client-side arithmetic is the
  same one the whole product is built on — a weight multiplied by the user's capital —
  and the count-up animates the DISPLAY of a figure that is already decided, landing
  exactly on it. Nothing here is a projection: each stat is labelled as a historical
  characterisation of today's weights, which is what it is.
*/

import { money, pct, num, beta as fmtBeta } from "../lib/format";
import { useCountUp } from "./useCountUp";
import { Badge, Tooltip } from "./ui";

/** One hero stat. `raw` is the real value; `render` turns the animated number into text. */
function Stat({ label, raw, render, tone, sub, tip, delay }) {
  const shown = useCountUp(raw, { delay });
  return (
    <div className="hero-stat">
      <div className="hero-stat-k">
        {tip ? <Tooltip text={tip}>{label}</Tooltip> : label}
      </div>
      <div className={`hero-stat-v num${tone ? ` ${tone}` : ""}`}>{render(shown)}</div>
      {sub && <div className="hero-stat-sub">{sub}</div>}
    </div>
  );
}

export default function PieHero({ index, pie, capital }) {
  const sc = pie.scorecard || {};
  const invested = (1 - pie.cash_weight) * capital;

  return (
    <header className="hero ra-rise">
      <div className="hero-head">
        <div>
          <div className="hero-eyebrow">
            <Badge>Educational simulation</Badge>
            <span className="faint">Data as of {pie.as_of}</span>
          </div>
          <h1 className="hero-title">
            {money(capital, index.currency)} becomes a{" "}
            <strong className="num">{pie.n_holdings}</strong>-stock pie
            {pie.cash_weight > 0.0005 && (
              <> with <strong className="num">{pct(pie.cash_weight, 0)}</strong> held in cash</>
            )}.
          </h1>
          <p className="hero-sub">
            Built by the frozen ranker for a target of{" "}
            <span className="num">β{fmtBeta(pie.target_beta)}</span> — the risk level you
            choose below. Every number here is a historical characterisation of these
            weights, never a forecast.
          </p>
        </div>
      </div>

      <div className="hero-stats">
        <Stat
          label="Invested"
          raw={invested}
          delay={0}
          render={(v) => money(v, index.currency)}
          sub={`across ${pie.n_holdings} stocks`}
        />
        <Stat
          label="Cash sleeve"
          raw={pie.cash_weight}
          delay={60}
          render={(v) => pct(v)}
          tip="Cash is how the engine lowers risk — your shock absorber. It has a beta of zero, so the more cash you hold, the less your pie moves with the market."
          sub={money(pie.cash_weight * capital, index.currency)}
        />
        <Stat
          label="Realised beta"
          raw={pie.achieved_beta}
          delay={120}
          render={(v) => fmtBeta(v)}
          tip="Beta is how much your pie moves when the market moves. Beta 0.5 means roughly half the market's swing."
          sub={`target ${fmtBeta(pie.target_beta)}`}
        />
        <Stat
          label="Sharpe"
          raw={sc.sharpe}
          delay={180}
          render={(v) => num(v)}
          tip="Return per unit of risk, measured over the committed history. Higher is better, but it says nothing about how the fall felt."
          sub="return per unit of risk"
        />
        <Stat
          label="Max drawdown"
          raw={sc.max_drawdown}
          delay={240}
          render={(v) => pct(v)}
          tone="loss"
          tip="The worst peak-to-trough fall over the measured history — the thing that actually tests whether you can hold a position."
          sub="worst peak-to-trough fall"
        />
      </div>

      {pie.beta_capped && (
        <p className="hero-note loss">
          Target β{fmtBeta(pie.target_beta)} is not reachable without leverage — showing the
          highest achievable, β{fmtBeta(pie.achieved_beta)}.
        </p>
      )}
    </header>
  );
}
