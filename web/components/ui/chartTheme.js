/*
  RankAlpha chart theme — Frontend v3, WP1.

  One place that decides what colour a mark is, so two charts built in two different work
  packages cannot disagree about what "the pie" or "the benchmark" looks like.

  Pure data + pure functions on purpose: no JSX, no React, no DOM. That makes the rules
  below unit-testable (chartTheme.test.js) rather than assertions in a comment.

  ── How these colours were chosen ────────────────────────────────────────────────────
  Not by eye. Both ramps were generated in OKLCH and run through the data-viz palette
  validator against THIS product's chart surface (--ra-surface #12171f), dark mode:

    Categorical (7 slots, adjacent pairlist)
      Lightness band       PASS  all 7 inside L 0.48–0.67
      Chroma floor         PASS  all 7 >= 0.10
      CVD separation       PASS  worst adjacent magenta↔yellow ΔE 13.2 (deutan)
      Normal-vision floor  PASS  worst adjacent magenta↔yellow ΔE 19.3
      Contrast vs surface  PASS  all 7 >= 3:1

    Accent ramp (8 steps, --ordinal)
      Lightness monotone   PASS
      Adjacent ΔL          PASS  all gaps >= 0.06
      Light-end contrast   PASS  darkest step 2.22:1 vs surface
      Single hue           PASS  hue spread 1°

  The slot ORDER is the colourblind-safety mechanism, not a mood choice: all 720
  orderings of slots 2–7 were enumerated with slot 1 pinned to the signature hue, 100
  cleared every gate, and this is the one that maximises the worst adjacent pair.

  ── Two rules that are not negotiable ────────────────────────────────────────────────
  1. RED IS NOT IN THIS PALETTE. It is reserved for losses and warnings. A series that
     could wear red could be mistaken for a loss, so the categorical set is 7 hues, not
     the usual 8, and `LOSS` lives in its own export.
  2. HUES ARE NEVER CYCLED. `seriesColor` returns null past the last slot rather than
     wrapping around, because a wrapped 8th series is a lie about identity. Past 7,
     fold the tail into "Other", facet the chart, or switch to the magnitude ramp.
*/

/** The surface these ramps were validated against. Re-run the validator if it changes. */
export const CHART_SURFACE = "#12171f";

/**
 * Categorical slots — identity ("which series is this"). Fixed order, assigned in
 * sequence from slot 1, never reordered per chart and never cycled.
 */
export const CATEGORICAL = [
  { slot: 1, hue: "teal", hex: "#23a897" },   // the signature hue, stepped for the dark surface
  { slot: 2, hue: "orange", hex: "#d95926" },
  { slot: 3, hue: "blue", hex: "#3987e5" },
  { slot: 4, hue: "yellow", hex: "#c98500" },
  { slot: 5, hue: "magenta", hex: "#d55181" },
  { slot: 6, hue: "violet", hex: "#9085e9" },
  { slot: 7, hue: "green", hex: "#008300" },
];

/**
 * How many slots survive the harder all-pairs test — the one that applies when any two
 * marks can end up side by side (scatter, bubble, small multiples). Measured, not
 * guessed: slots 1–3 pass, slot 4 puts yellow next to orange and fails the
 * normal-vision floor at ΔE 10.6. Those chart forms carry three series, then "Other".
 */
export const CATEGORICAL_ALL_PAIRS_CAP = 3;

/**
 * Magnitude ramp — one hue, monotone lightness, brightest first. This is what a
 * weight-ordered mark uses (the donut, a weight bar), because portfolio weight is a
 * magnitude, not an identity: the biggest holding should LOOK biggest, and a 20-holding
 * pie has no honest categorical answer.
 */
export const ACCENT_RAMP = [
  "#38edd6", "#32d6c1", "#2bc0ad", "#25ab9a",
  "#1e9686", "#188174", "#126d61", "#0a5a50",
];

/** Reserved, non-series colours. Each has exactly one meaning. */
export const LOSS = "#ff6b6b";        // losses and warnings only
export const CASH = "#69778a";        // the cash slice — first-class, deliberately quiet
export const BENCHMARK = "#8592a3";   // "not us": muted ink plus a dash, never a hue
export const NUM_NEUTRAL = "#c3cbd8"; // a delta that is information, not a loss

/** Chart chrome, matching the CSS classes in styles/ui.css. */
export const CHROME = {
  grid: "var(--ra-grid)",
  axis: "var(--ra-axis)",
  axisInk: "var(--ra-axis-ink)",
  seriesWidth: 2,
  benchmarkWidth: 1.5,
  markerMin: 8,      // px — the smallest a hoverable point may be
  fillGap: 2,        // px of surface between adjacent fills, so marks never merge
  pad: { t: 12, r: 10, b: 24, l: 48 },
};

/**
 * The colour of categorical series `i` (0-based).
 *
 * Returns null past the last slot — deliberately. A caller that hits null has more
 * identities than the palette can safely distinguish and must fold, facet, or switch
 * encoding. Cycling back to slot 1 would silently give two series the same colour.
 */
export function seriesColor(i) {
  if (!Number.isInteger(i) || i < 0) return null;
  return i < CATEGORICAL.length ? CATEGORICAL[i].hex : null;
}

/**
 * A step of the magnitude ramp for item `i` of `n`, ordered largest-first.
 *
 * With n <= 8 each item gets its own step, so neighbours differ by at least ΔL 0.06.
 * With n > 8 the ramp is read as continuous magnitude and nearby items may share a
 * step — correct for a sequential encoding, and the reason every chart using this must
 * also carry direct labels or a legend (see `needsLegend`).
 */
export function magnitudeColor(i, n) {
  if (!Number.isInteger(i) || i < 0 || !Number.isInteger(n) || n < 1) return null;
  if (i >= n) return null;
  if (n === 1) return ACCENT_RAMP[0];
  const t = i / (n - 1);
  return ACCENT_RAMP[Math.round(t * (ACCENT_RAMP.length - 1))];
}

/** A legend is mandatory from two series up — identity is never colour alone. */
export const needsLegend = (nSeries) => nSeries >= 2;

/**
 * Direct labels are how a magnitude-encoded chart stays readable when steps repeat, and
 * how a sub-3:1 mark earns its place. Charts ask this rather than deciding locally.
 */
export const needsDirectLabels = (nMarks) => nMarks > ACCENT_RAMP.length;

/**
 * "Nice" axis ticks: at most `count` round values covering [lo, hi].
 *
 * Every chart in the product shares this so gridlines land on the same kind of number
 * whatever the panel. It computes tick POSITIONS only — never a statistic, so there is
 * no path by which a chart can show a figure the exporter did not produce.
 */
export function niceTicks(lo, hi, count = 4) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || count < 2) return [];
  if (lo === hi) return [lo];
  if (lo > hi) [lo, hi] = [hi, lo];
  const raw = (hi - lo) / (count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  // Geometric-midpoint thresholds (√2, √10, √50) rather than 1/2/5: they snap to whichever
  // round step is actually closest, so asking for 5 ticks over [0,1] gives 0.2 and not the
  // 0.5 a linear cut-off would round up to.
  const step = (norm < Math.SQRT2 ? 1 : norm < Math.sqrt(10) ? 2 : norm < Math.sqrt(50) ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    // Re-round: repeated addition of a float step drifts (0.1+0.2…), and a tick label
    // reading "0.30000000000000004" would be a number the UI invented.
    out.push(Number(v.toFixed(10)));
  }
  return out;
}
