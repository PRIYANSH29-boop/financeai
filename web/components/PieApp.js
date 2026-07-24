"use client";

/*
  The single page. Holds the two pieces of user state (capital, target beta) and swaps in
  the precomputed pie for the selected beta.

  Data rule: the initial pie is embedded at build time so first paint has real numbers and
  never flashes a placeholder; every other beta is fetched from /bundle/beta/<key>.json —
  a static file on the same origin, which keeps the export backend-free. Fetched pies are
  memoised so dragging the slider back and forth does not refetch.
*/

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Donut from "./Donut";
import {
  ControlBar, WhyHolding, DriftPanel, HonestyPanel, HowItWorks,
} from "./Panels";

export default function PieApp({ index, initialPie }) {
  const [capital, setCapital] = useState(index.default_capital);
  const [target, setTarget] = useState(initialPie.target_beta);
  const [pie, setPie] = useState(initialPie);
  const [selected, setSelected] = useState(null);
  const [showHow, setShowHow] = useState(false);
  const [error, setError] = useState(null);

  const cache = useRef({ [initialPie.target_beta]: initialPie });

  const keyFor = useCallback(
    (b) => index.beta_keys[b.toFixed(2)],
    [index.beta_keys]
  );

  useEffect(() => {
    let cancelled = false;
    const cached = cache.current[target];
    if (cached) { setPie(cached); setError(null); return; }

    const key = keyFor(target);
    if (!key) {
      // No bundle entry: snap, never interpolate. Inventing a pie between two grid points
      // would put numbers on screen that the engine never produced.
      setError("No precomputed pie for that risk level.");
      return;
    }
    fetch(`bundle/beta/${key}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data) => {
        if (cancelled) return;
        cache.current[target] = data;
        setPie(data);
        setError(null);
      })
      .catch((e) => !cancelled && setError(`Could not load that risk level (${e.message}).`));
    return () => { cancelled = true; };
  }, [target, keyFor]);

  // A holding can drop out of the book when the beta target changes; clear a stale
  // selection rather than leaving zone 3 describing something no longer in the pie.
  useEffect(() => {
    if (selected && selected !== "CASH" &&
        !pie.holdings.some((h) => h.ticker === selected)) {
      setSelected(null);
    }
  }, [pie, selected]);

  const asOf = useMemo(() => pie.as_of, [pie]);

  return (
    <>
      <header className="site-header">
        <div className="inner">
          <span className="brand">Rank<span>Alpha</span></span>
          <span className="sim-badge">Educational simulation</span>
          <button className="link-btn" onClick={() => setShowHow(true)}>
            How it works →
          </button>
        </div>
      </header>

      <main className="wrap">
        <div className="stack" style={{ paddingTop: 22 }}>
          <ControlBar index={index} capital={capital} setCapital={setCapital}
                      target={target} setTarget={setTarget} pie={pie} />

          {error && (
            <div className="card" role="alert">
              <span className="loss">{error}</span>
            </div>
          )}

          <div className="main-grid">
            <div className="card">
              <h2>Your pie</h2>
              <Donut holdings={pie.holdings} cashWeight={pie.cash_weight}
                     capital={capital} currency={index.currency}
                     selected={selected} onSelect={setSelected} />
              <div className="guardrails">{index.guardrails.text}</div>
            </div>

            <WhyHolding pie={pie} selected={selected} capital={capital}
                        currency={index.currency} guardrails={index.guardrails} />
          </div>

          <DriftPanel pie={pie} />
          <HonestyPanel pie={pie} index={index} capital={capital} />

          <p className="faint" style={{ fontSize: "0.8rem", margin: 0 }}>
            Pie built from data as of {asOf}. Benchmark: {index.benchmark.label}.
          </p>
        </div>
      </main>

      <footer className="site-footer">
        <div className="inner">
          <span>Educational simulation. No real money. Not investment advice.</span>
          <a href="https://github.com/PRIYANSH29-boop/financeai" target="_blank"
             rel="noreferrer">GitHub repo</a>
          <span style={{ marginLeft: "auto" }}>Built by Priyansh Patel</span>
        </div>
      </footer>

      {showHow && (
        <HowItWorks steps={index.how_it_works} onClose={() => setShowHow(false)} />
      )}
    </>
  );
}
