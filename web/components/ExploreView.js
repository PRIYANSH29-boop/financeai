"use client";

/*
  Module A — Explore the market (Phase 23). An honest snapshot of the 1,200-name wide
  universe: searchable, sortable, cap-filterable, with winners/losers cards. Every figure is
  "as of" the bundle data date and stamped as such — never presented as live/today.

  Only model-scored (S&P 500) names are basket-eligible; those are marked, and the rest are
  visibly flagged as browse-only.

  Phase 24 — rows whose risk stats fail the exporter's sanity band (|beta| > 3 or ann vol >
  100%) carry stat_quality "unreliable". We show the measured numbers unchanged, mark them ⚠,
  and sort them LAST in the risk orderings: sorting by beta used to put a bankruptcy-splice
  artifact (CHRD, beta 364) at the top of the screen, which is the first thing a visitor saw.
*/

import { useMemo, useState } from "react";
import { pct, num, beta as fmtBeta } from "../lib/format";
import { sortRows } from "../lib/explore";

const MAX_ROWS = 100;   // keep the DOM light; the count line reports how many matched

const COLS = [
  { id: "ticker", label: "Ticker", numeric: false },
  { id: "name", label: "Name", numeric: false },
  { id: "sector", label: "Sector", numeric: false },
  { id: "cap_bucket", label: "Cap", numeric: false },
  { id: "last_return", label: "Last mo", numeric: true },
  { id: "ann_vol", label: "Ann vol", numeric: true },
  { id: "beta", label: "Beta", numeric: true },
];

function MoverCard({ title, rows, tone }) {
  return (
    <div className="card mover-card">
      <h3>{title}</h3>
      <ul className="mover-list">
        {rows.map((r) => (
          <li key={r.ticker}>
            <span className="num mover-tk">{r.ticker}</span>
            <span className="muted mover-nm">{r.name}</span>
            <span className={`num ${tone}`}>{pct(r.last_return, 1)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ExploreView({ explore, onPickBasket }) {
  const [q, setQ] = useState("");
  const [cap, setCap] = useState("all");        // all | mid | large
  const [scoredOnly, setScoredOnly] = useState(false);
  const [sort, setSort] = useState({ col: "last_return", dir: -1 });

  const filtered = useMemo(() => {
    if (!explore) return [];
    const needle = q.trim().toLowerCase();
    let rows = explore.rows.filter((r) => {
      if (cap !== "all" && r.cap_bucket !== cap) return false;
      if (scoredOnly && !r.scored) return false;
      if (needle && !(`${r.ticker} ${r.name}`.toLowerCase().includes(needle))) return false;
      return true;
    });
    // Ordering (incl. the #24 demotion of flagged rows) lives in lib/explore.js so it can be
    // unit-tested against the shipped bundle.
    return sortRows(rows, sort.col, sort.dir);
  }, [explore, q, cap, scoredOnly, sort]);

  const clickSort = (col, numeric) =>
    setSort((s) => (s.col === col ? { col, dir: -s.dir } : { col, dir: numeric ? -1 : 1 }));

  if (!explore) {
    return (
      <main className="wrap">
        <div className="card" style={{ marginTop: 22 }}>
          <p className="muted">
            The market explorer needs the wide-universe data, which isn&apos;t in this build.
            Run <code>make universe</code> then <code>make sectors</code> and rebuild the bundle.
          </p>
        </div>
      </main>
    );
  }

  const shown = filtered.slice(0, MAX_ROWS);
  const rec = explore.scored_reconciliation;

  return (
    <main className="wrap">
      <div className="stack" style={{ paddingTop: 22 }}>
        <div className="card">
          <div className="explore-head">
            <div>
              <h2 style={{ margin: 0 }}>Explore the market</h2>
              <p className="faint" style={{ margin: "6px 0 0" }}>
                Snapshot <strong>as of {explore.as_of}</strong> — not live. {explore.n_names}{" "}
                names · {explore.n_scored} basket-eligible
                {explore.n_unreliable ? (
                  <> · {explore.n_unreliable} ⚠ flagged unreliable</>
                ) : null}
                . Small caps (&lt;$2B) are excluded by universe methodology.
              </p>
              {rec ? (
                <details className="faint" style={{ marginTop: 8, fontSize: "0.8rem" }}>
                  <summary>Why {rec.basket_eligible} of {rec.model_universe} scored names?</summary>
                  <p style={{ margin: "6px 0 0" }}>
                    The frozen model ranks {rec.model_universe} S&amp;P 500 names.{" "}
                    {rec.with_explore_row} of them appear in this table;{" "}
                    {rec.dropped.absent_from_wide_universe} are absent from the wide-universe
                    screen entirely (a universe-construction bug, described below — they are
                    still reachable on the Basket tab).{" "}
                    {rec.dropped.failed_history_or_stats_on_sp_panel} more are refused for
                    insufficient or discontinuous history, and{" "}
                    {rec.dropped.failed_wide_panel_sanity_band} for stats outside the sanity
                    band, leaving {rec.basket_eligible}.
                  </p>
                  <p style={{ margin: "6px 0 0" }}>{rec.why_absent_from_wide_universe}</p>
                </details>
              ) : null}
            </div>
          </div>

          <div className="movers-grid">
            <MoverCard title={`Top ${explore.movers.winners.length} — last month`}
                       rows={explore.movers.winners} tone="num-pos" />
            <MoverCard title={`Bottom ${explore.movers.losers.length} — last month`}
                       rows={explore.movers.losers} tone="loss" />
          </div>
        </div>

        <div className="card">
          <div className="explore-filters">
            <input className="explore-search" type="search" value={q}
                   placeholder="Search ticker or company…"
                   onChange={(e) => setQ(e.target.value)} />
            <div className="chips">
              {["all", "large", "mid"].map((c) => (
                <button key={c} className="chip" aria-pressed={cap === c}
                        onClick={() => setCap(c)}>
                  {c === "all" ? "All caps" : `${c}-cap`}
                </button>
              ))}
              <button className="chip" aria-pressed={scoredOnly}
                      onClick={() => setScoredOnly((v) => !v)}>
                Scored only
              </button>
            </div>
          </div>

          <div className="table-scroll">
            <table className="explore-table">
              <thead>
                <tr>
                  {COLS.map((c) => (
                    <th key={c.id} onClick={() => clickSort(c.id, c.numeric)}
                        className={c.numeric ? "num-col" : undefined}
                        aria-sort={sort.col === c.id ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
                      {c.label}{sort.col === c.id ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                    </th>
                  ))}
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => {
                  const shaky = r.stat_quality === "unreliable";
                  return (
                  <tr key={r.ticker} className={r.scored ? undefined : "not-scored"}>
                    <td className="num">
                      {r.ticker}
                      {shaky ? (
                        <span className="stat-warn" title={r.stat_note}> ⚠</span>
                      ) : null}
                    </td>
                    <td className="cell-name">{r.name}</td>
                    <td><span className="sector-tag">{r.sector}</span></td>
                    <td>{r.cap_bucket}</td>
                    <td className={`num num-col ${r.last_return >= 0 ? "num-pos" : "loss"}`}>
                      {pct(r.last_return, 1)}
                    </td>
                    <td className={`num num-col${shaky ? " stat-shaky" : ""}`}
                        title={shaky ? r.stat_note : undefined}>
                      {pct(r.ann_vol, 0)}
                    </td>
                    <td className={`num num-col${shaky ? " stat-shaky" : ""}`}
                        title={shaky ? r.stat_note : undefined}>
                      {fmtBeta(r.beta)}
                    </td>
                    <td>
                      {r.scored ? (
                        <button className="chip tiny" onClick={onPickBasket}
                                title="Model-scored — usable in Build your basket">
                          basket →
                        </button>
                      ) : (
                        <span className="faint tiny"
                              title={r.scored_reason || "Not scored by the frozen model"}>
                          browse only
                        </span>
                      )}
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="faint" style={{ fontSize: "0.8rem", marginTop: 10 }}>
            Showing {shown.length} of {filtered.length} matching names
            {filtered.length > MAX_ROWS ? " — refine the search to see more." : "."}{" "}
            Beta &amp; vol measured over the committed history vs the equal-weight universe.
            {explore.sanity_band ? ` ${explore.sanity_band.text}` : ""}
          </div>
        </div>
      </div>
    </main>
  );
}
