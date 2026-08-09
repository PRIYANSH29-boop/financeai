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
import PieHero from "./PieHero";
import { DonutSkeleton, ErrorState } from "./States";
import {
  ControlBar, WhyHolding, DriftPanel, HonestyPanel,
} from "./Panels";

export default function PieApp({ index, initialPie }) {
  const [capital, setCapital] = useState(index.default_capital);
  const [target, setTarget] = useState(initialPie.target_beta);
  const [pie, setPie] = useState(initialPie);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [attempt, setAttempt] = useState(0);

  const cache = useRef({ [initialPie.target_beta]: initialPie });

  const keyFor = useCallback(
    (b) => index.beta_keys[b.toFixed(2)],
    [index.beta_keys]
  );

  useEffect(() => {
    let cancelled = false;
    const cached = cache.current[target];
    if (cached) { setPie(cached); setError(null); setLoading(false); return; }

    const key = keyFor(target);
    if (!key) {
      // No bundle entry: snap, never interpolate. Inventing a pie between two grid points
      // would put numbers on screen that the engine never produced.
      setError({ title: "No precomputed pie for that risk level.",
                 detail: "The slider snaps to the levels the engine actually built. Pick the nearest one." });
      setLoading(false);
      return;
    }
    setLoading(true);
    fetch(`bundle/beta/${key}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data) => {
        if (cancelled) return;
        cache.current[target] = data;
        setPie(data);
        setError(null);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError({
          title: "Could not load that risk level.",
          // The real reason, not a generic apology — the reader may be the only person who
          // can act on it (offline, blocked, a half-finished deploy).
          detail: `${e.message}. The figures below are still the ones for β${pie.target_beta.toFixed(2)}, not the level you just picked.`,
        });
        setLoading(false);
      });
    return () => { cancelled = true; };
    // `pie` is read only to name the stale beta in the error copy; depending on it would
    // refetch every time a pie lands.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, keyFor, attempt]);

  // Re-running the effect needs a value that actually changes: setTarget(b => b) is a
  // no-op React bails out of, so a retry on the same beta would do nothing.
  const retry = useCallback(() => {
    delete cache.current[target];
    setError(null);
    setAttempt((n) => n + 1);
  }, [target]);

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
    <main>
      {/* The result first. The controls that produced it come immediately after, so the
          page opens with an answer rather than a form. */}
      <PieHero index={index} pie={pie} capital={capital} />

      <div className="wrap">
        <div className="stack">
          <ControlBar index={index} capital={capital} setCapital={setCapital}
                      target={target} setTarget={setTarget} pie={pie} />

          {error && (
            <div className="card">
              <ErrorState title={error.title} detail={error.detail} onRetry={retry} />
            </div>
          )}

          <div className="main-grid">
            <div className="card">
              {/* The beta badges that used to sit here now lead the hero, so they are not
                  repeated. What this header adds instead is what the chart is FOR. */}
              <div className="pie-head">
                <h2 style={{ margin: 0 }}>Your pie</h2>
                <span className="faint" style={{ fontSize: "0.8rem" }}>
                  Click a slice to see why it&apos;s held
                </span>
              </div>
              {loading ? (
                <DonutSkeleton />
              ) : (
                <Donut holdings={pie.holdings} cashWeight={pie.cash_weight}
                       capital={capital} currency={index.currency}
                       selected={selected} onSelect={setSelected} />
              )}
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
      </div>
    </main>
  );
}
