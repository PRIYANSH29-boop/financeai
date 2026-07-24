/*
  node:test suite for the client-side basket math (lib/basket.js). Runs with `npm test`
  (node --test) — zero dependencies, built into Node 18+. Tests the REAL functions the
  Basket page uses, against hand-checked fixtures that mirror the Python analytics.metrics
  reference (so JS and Python agree), plus the rolling-12mo distribution and the as-of stamp.
*/

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  std, annVol, sharpe, sortino, maxDrawdown, beta, totalReturn,
  basketSeries, basketScorecard, rolling12mReturns, median, outcomeDistribution,
} from "./basket.js";

const approx = (a, b, eps = 1e-9) =>
  assert.ok(Math.abs(a - b) <= eps, `${a} !== ${b} (±${eps})`);

// ---- population std (ddof=0) matches the committed Python hand-check vol([.12,-.08,.04,.08]) = 7.48%
test("std ddof=0 matches the analytics.metrics hand-check", () => {
  approx(std([0.12, -0.08, 0.04, 0.08]), 0.0748331477, 1e-9);
});

test("annVol annualises by sqrt(12)", () => {
  approx(annVol([0.12, -0.08, 0.04, 0.08]), 0.0748331477 * Math.sqrt(12), 1e-9);
});

// ---- beta hand-check: a series that is exactly 2x the benchmark has beta 2.0
test("beta hand-check: port = 2x bench -> 2.0", () => {
  const bench = [0.02, -0.03, 0.01, 0.04];
  const port = bench.map((x) => 2 * x);
  approx(beta(port, bench), 2.0, 1e-12);
  approx(beta(bench, bench), 1.0, 1e-12);
});

// ---- sharpe reference: mean/std * sqrt(12)
test("sharpe = mean/std * sqrt(12)", () => {
  const r = [0.02, 0.01, -0.01, 0.03];
  const m = r.reduce((s, x) => s + x, 0) / r.length;
  approx(sharpe(r), (m / std(r)) * Math.sqrt(12), 1e-12);
});

test("sortino uses downside deviation over the full n", () => {
  const r = [0.02, -0.03, 0.01, 0.04];
  const dd = Math.sqrt([0, -0.03, 0, 0].map((x) => x * x).reduce((s, x) => s + x, 0) / 4);
  const m = r.reduce((s, x) => s + x, 0) / 4;
  approx(sortino(r), (m / dd) * Math.sqrt(12), 1e-12);
});

test("maxDrawdown from a known path", () => {
  // equity: 1 -> 1.1 -> 0.99 -> ... worst dd from peak 1.1 to 0.99 = -0.10
  approx(maxDrawdown([0.1, -0.1]), 0.99 / 1.1 - 1, 1e-12);
});

test("totalReturn compounds", () => {
  approx(totalReturn([0.1, -0.1]), 1.1 * 0.9 - 1, 1e-12);
});

// ---- equal-weight basket series: mean across picks per month, skipping nulls
test("basketSeries equal-weights across picks and skips null months", () => {
  const stocks = {
    dates: ["2020-01-31", "2020-02-29", "2020-03-31"],
    benchmark_returns: [0.01, 0.02, 0.03],
    returns: { A: [0.10, null, 0.04], B: [0.20, 0.06, null] },
  };
  const { dates, basket, bench, nPicks } = basketSeries(["A", "B"], stocks);
  assert.equal(nPicks, 2);
  assert.deepEqual(dates, ["2020-01-31", "2020-02-29", "2020-03-31"]);
  approx(basket[0], 0.15, 1e-12);   // (0.10+0.20)/2
  approx(basket[1], 0.06, 1e-12);   // only B has data
  approx(basket[2], 0.04, 1e-12);   // only A has data
  assert.deepEqual(bench, [0.01, 0.02, 0.03]);
});

// ---- rolling 12-month distribution, hand-checked on a constant fixture
test("rolling12mReturns: 14 constant months -> 3 windows, all equal", () => {
  const r = Array(14).fill(0.01);
  const roll = rolling12mReturns(r, 12);
  assert.equal(roll.length, 3);                 // windows [0..11],[1..12],[2..13]
  const expect = Math.pow(1.01, 12) - 1;
  for (const x of roll) approx(x, expect, 1e-12);
});

test("median: odd and even lengths", () => {
  approx(median([3, 1, 2]), 2, 1e-12);
  approx(median([4, 1, 3, 2]), 2.5, 1e-12);
});

test("outcomeDistribution returns a RANGE (min<=median<=max), never a point forecast", () => {
  // ramping returns so windows differ
  const stocks = {
    dates: Array.from({ length: 15 }, (_, i) => `2020-${String(i + 1).padStart(2, "0")}`),
    benchmark_returns: Array(15).fill(0),
    returns: { A: Array.from({ length: 15 }, (_, i) => 0.005 * (i + 1)) },
  };
  const d = outcomeDistribution(["A"], stocks, 12);
  assert.equal(d.n, 4);                          // 15 - 12 + 1
  assert.ok(d.min <= d.median && d.median <= d.max);
  assert.equal(d.window, 12);
  // the object exposes ONLY a range — no expected/forecast/point field
  assert.deepEqual(Object.keys(d).sort(), ["max", "median", "min", "n", "window"]);
});

test("basketScorecard produces analyser-convention fields and no forward field", () => {
  const stocks = {
    dates: ["a", "b", "c", "d"],
    benchmark_returns: [0.01, -0.01, 0.02, 0.0],
    returns: { A: [0.02, -0.02, 0.03, 0.01], B: [0.00, 0.00, 0.01, -0.01] },
  };
  const sc = basketScorecard(["A", "B"], stocks);
  assert.equal(sc.n_months, 4);
  for (const k of ["total_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "beta", "hit_rate"])
    assert.ok(k in sc, `missing ${k}`);
  assert.ok(!("expected_return" in sc) && !("forecast" in sc) && !("projection" in sc));
});

// ---- the shipped bundle carries an as-of stamp (guards the "never present as live" rule)
test("stocks.json bundle has an as_of stamp and aligned series", () => {
  const p = "../public/bundle/stocks.json";
  let s;
  try { s = JSON.parse(readFileSync(new URL(p, import.meta.url))); }
  catch { return; }   // bundle not built in this checkout — skip rather than fail
  assert.ok(s.as_of, "stocks.json missing as_of");
  const n = s.dates.length;
  assert.equal(s.benchmark_returns.length, n);
  for (const tk of Object.keys(s.returns)) assert.equal(s.returns[tk].length, n);
});
