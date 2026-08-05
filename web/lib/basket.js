/*
  Client-side basket math for Module B — a faithful mirror of Python `analytics.metrics`
  (population std, ddof=0; PPY=12 monthly). Every number the Basket page shows is computed
  here from the per-stock monthly series shipped in stocks.json. Pure functions, no DOM, so
  `node:test` can verify them against the same hand-checked fixtures the Python side uses.

  HARD RULE (#23): there is deliberately NO expected-return / forward-projection function in
  this file. The only answer to "what will I get?" is the historical distribution RANGE.
*/

const PPY = 12;

export const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN);

// Population standard deviation (ddof=0) — matches analytics.metrics.volatility.
export function std(a) {
  if (a.length === 0) return NaN;
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
}

export const annVol = (r) => std(r) * Math.sqrt(PPY);

export function totalReturn(r) {
  return r.reduce((acc, x) => acc * (1 + x), 1) - 1;
}

export function sharpe(r) {
  const sd = std(r);
  return sd === 0 || Number.isNaN(sd) ? NaN : (mean(r) / sd) * Math.sqrt(PPY);
}

// Downside deviation over the same n (ddof=0), matching analytics.metrics.sortino.
export function sortino(r) {
  if (r.length === 0) return NaN;
  const dn = r.map((x) => (x < 0 ? x : 0));
  const dd = Math.sqrt(mean(dn.map((x) => x * x)));
  return dd === 0 ? NaN : (mean(r) / dd) * Math.sqrt(PPY);
}

// Max drawdown from a return series (builds the equity curve first). Returns a negative number.
export function maxDrawdown(r) {
  let eq = 1, peak = 1, mdd = 0;
  for (const x of r) {
    eq *= 1 + x;
    if (eq > peak) peak = eq;
    const dd = eq / peak - 1;
    if (dd < mdd) mdd = dd;
  }
  return mdd;
}

// Cov(r, b) / Var(b), population — matches analytics.metrics.beta. Pairs must align by index.
export function beta(r, b) {
  const n = Math.min(r.length, b.length);
  if (n < 2) return NaN;
  const rr = r.slice(0, n), bb = b.slice(0, n);
  const mr = mean(rr), mb = mean(bb);
  const varB = mean(bb.map((x) => (x - mb) ** 2));
  if (varB === 0) return NaN;
  const cov = mean(rr.map((x, i) => (x - mr) * (bb[i] - mb)));
  return cov / varB;
}

export const hitRate = (r) => (r.length ? r.filter((x) => x > 0).length / r.length : NaN);

/*
  Equal-weight basket monthly returns from the shipped series.
  `picks`   : array of tickers.
  `stocks`  : the stocks.json payload ({dates, returns:{tk:[...]}, benchmark_returns}).
  Returns { dates, basket, bench } filtered to months where AT LEAST ONE pick has data;
  each basket month = equal-weight mean over the picks that have a value that month.
*/
export function basketSeries(picks, stocks) {
  const { dates, returns, benchmark_returns: bench } = stocks;
  // Dedupe first (#25 B-8). A repeated ticker used to be counted twice: `nPicks` over-stated
  // the basket, and — worse — with one duplicate plus any other name the repeated name got
  // double weight in the equal-weight mean, so an "equal-weight basket" quietly wasn't one.
  const valid = [...new Set(picks)].filter((tk) => Array.isArray(returns[tk]));
  const outDates = [], outBasket = [], outBench = [];
  for (let i = 0; i < dates.length; i++) {
    const vals = valid.map((tk) => returns[tk][i]).filter((v) => v !== null && v !== undefined);
    if (vals.length === 0) continue;                 // no pick has data this month → skip
    outDates.push(dates[i]);
    outBasket.push(mean(vals));                       // equal-weight
    outBench.push(bench[i]);
  }
  return { dates: outDates, basket: outBasket, bench: outBench, nPicks: valid.length };
}

// Full scorecard for an equal-weight basket — analyser conventions, past tense only.
export function basketScorecard(picks, stocks) {
  const { basket, bench, dates } = basketSeries(picks, stocks);
  return {
    n_months: basket.length,
    start: dates[0] || null,
    end: dates[dates.length - 1] || null,
    total_return: totalReturn(basket),
    ann_vol: annVol(basket),
    sharpe: sharpe(basket),
    sortino: sortino(basket),
    max_drawdown: maxDrawdown(basket),
    beta: beta(basket, bench),
    hit_rate: hitRate(basket),
  };
}

/*
  Every rolling 12-month COMPOUNDED basket return over the committed history. This is the
  ONLY permitted answer to the horizon question — a distribution of what a 12-month hold
  WOULD HAVE returned, never a forward projection.
*/
export function rolling12mReturns(basketMonthly, window = 12) {
  const out = [];
  for (let i = 0; i + window <= basketMonthly.length; i++) {
    let acc = 1;
    for (let j = i; j < i + window; j++) acc *= 1 + basketMonthly[j];
    out.push(acc - 1);
  }
  return out;
}

export function median(a) {
  if (a.length === 0) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// { min, median, max, n } of the rolling-12mo distribution — the RANGE the UI shows.
export function outcomeDistribution(picks, stocks, window = 12) {
  const { basket } = basketSeries(picks, stocks);
  const roll = rolling12mReturns(basket, window);
  if (roll.length === 0) return { n: 0, min: null, median: null, max: null, window };
  return {
    n: roll.length,
    min: Math.min(...roll),
    median: median(roll),
    max: Math.max(...roll),
    window,
  };
}
