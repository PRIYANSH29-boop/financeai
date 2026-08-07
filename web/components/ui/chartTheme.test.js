/*
  Chart-theme rules, as tests.

  The palette gates themselves (colour-vision ΔE, lightness band, contrast) were run with
  the data-viz validator when the ramps were generated — that tool is not a dependency of
  this repo, and vendoring it to re-derive a constant would be theatre. What IS worth
  guarding here is everything a future edit could quietly break: the reserved colours
  staying out of the series palette, the no-cycling rule, ramp monotonicity, and the tick
  generator never inventing a value.
*/

import test from "node:test";
import assert from "node:assert/strict";
import {
  CATEGORICAL, CATEGORICAL_ALL_PAIRS_CAP, ACCENT_RAMP, LOSS, CASH, BENCHMARK,
  seriesColor, magnitudeColor, needsLegend, needsDirectLabels, niceTicks,
} from "./chartTheme.js";

/* ------------------------------------------------------------------ reserved colours */

test("red is not a series colour — it is reserved for losses", () => {
  const hexes = CATEGORICAL.map((c) => c.hex.toLowerCase());
  assert.ok(!hexes.includes(LOSS.toLowerCase()));
  // and no slot is even named red: the exclusion is structural, not a near-miss
  assert.ok(!CATEGORICAL.some((c) => c.hue === "red"));
});

test("cash and benchmark are reserved too — neither can be handed out as a series", () => {
  const hexes = CATEGORICAL.map((c) => c.hex.toLowerCase());
  assert.ok(!hexes.includes(CASH.toLowerCase()));
  assert.ok(!hexes.includes(BENCHMARK.toLowerCase()));
});

test("every categorical slot is a distinct hex", () => {
  const hexes = CATEGORICAL.map((c) => c.hex.toLowerCase());
  assert.equal(new Set(hexes).size, hexes.length);
});

test("slots are numbered 1..n in order — the order IS the safety mechanism", () => {
  CATEGORICAL.forEach((c, i) => assert.equal(c.slot, i + 1));
});

test("slot 1 is the signature hue", () => {
  assert.equal(CATEGORICAL[0].hue, "teal");
});

/* ------------------------------------------------------------------ no cycling */

test("seriesColor hands out slots in fixed order", () => {
  CATEGORICAL.forEach((c, i) => assert.equal(seriesColor(i), c.hex));
});

test("seriesColor returns null past the last slot instead of wrapping", () => {
  assert.equal(seriesColor(CATEGORICAL.length), null);
  assert.equal(seriesColor(CATEGORICAL.length + 5), null);
  // the failure mode this guards: an 8th series silently sharing slot 1's colour
  assert.notEqual(seriesColor(0), seriesColor(CATEGORICAL.length));
});

test("seriesColor rejects nonsense indices rather than coercing them", () => {
  assert.equal(seriesColor(-1), null);
  assert.equal(seriesColor(1.5), null);
  assert.equal(seriesColor("2"), null);
});

test("the all-pairs cap is smaller than the adjacent-pairs palette", () => {
  // scatter/bubble/small-multiples can put any two marks side by side, so fewer slots
  // survive there. If this ever equals the full length, someone dropped the harder gate.
  assert.ok(CATEGORICAL_ALL_PAIRS_CAP >= 1);
  assert.ok(CATEGORICAL_ALL_PAIRS_CAP < CATEGORICAL.length);
});

/* ------------------------------------------------------------------ magnitude ramp */

test("the magnitude ramp is a single monotone lightness sequence, brightest first", () => {
  const lum = (hex) => {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  for (let i = 1; i < ACCENT_RAMP.length; i++) {
    assert.ok(lum(ACCENT_RAMP[i]) < lum(ACCENT_RAMP[i - 1]),
              `step ${i} is not darker than step ${i - 1}`);
  }
});

test("magnitudeColor gives the brightest step to the biggest item", () => {
  assert.equal(magnitudeColor(0, 8), ACCENT_RAMP[0]);
  assert.equal(magnitudeColor(7, 8), ACCENT_RAMP[ACCENT_RAMP.length - 1]);
  assert.equal(magnitudeColor(0, 1), ACCENT_RAMP[0]);
});

test("magnitudeColor gives every item its own step when they fit", () => {
  const n = ACCENT_RAMP.length;
  const used = Array.from({ length: n }, (_, i) => magnitudeColor(i, n));
  assert.equal(new Set(used).size, n);
});

test("magnitudeColor stays inside the documented ramp when items outnumber steps", () => {
  // 24 holdings is a real pie size. Steps may repeat — that is a sequential encoding, and
  // it is why needsDirectLabels goes true — but no colour may be invented.
  for (let i = 0; i < 24; i++) {
    assert.ok(ACCENT_RAMP.includes(magnitudeColor(i, 24)));
  }
});

test("magnitudeColor refuses an index outside the set", () => {
  assert.equal(magnitudeColor(5, 5), null);
  assert.equal(magnitudeColor(-1, 5), null);
  assert.equal(magnitudeColor(0, 0), null);
});

/* ------------------------------------------------------------------ readability rules */

test("a legend is mandatory from two series up, and pointless below", () => {
  assert.equal(needsLegend(1), false);
  assert.equal(needsLegend(2), true);
  assert.equal(needsLegend(20), true);
});

test("direct labels become mandatory exactly when ramp steps start repeating", () => {
  assert.equal(needsDirectLabels(ACCENT_RAMP.length), false);
  assert.equal(needsDirectLabels(ACCENT_RAMP.length + 1), true);
});

/* ------------------------------------------------------------------ ticks */

test("niceTicks lands on round numbers inside the range", () => {
  const t = niceTicks(0, 80, 5);
  assert.deepEqual(t, [0, 20, 40, 60, 80]);
});

test("niceTicks never emits a float-drift artefact", () => {
  // 0 + 0.2 + 0.2 + 0.2 is 0.6000000000000001 in IEEE754, and an axis label reading that
  // would be a number the UI invented. The generator re-rounds each step.
  assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
});

test("niceTicks handles a reversed range and a degenerate one", () => {
  assert.deepEqual(niceTicks(80, 0, 5), niceTicks(0, 80, 5));
  assert.deepEqual(niceTicks(5, 5, 4), [5]);
});

test("niceTicks returns nothing rather than guessing on bad input", () => {
  assert.deepEqual(niceTicks(NaN, 10, 4), []);
  assert.deepEqual(niceTicks(0, Infinity, 4), []);
  assert.deepEqual(niceTicks(0, 10, 1), []);
});

test("niceTicks covers negative ranges — a drawdown axis is all negative", () => {
  const t = niceTicks(-0.4, 0, 5);
  assert.ok(t.includes(0));
  assert.ok(t.every((v) => v >= -0.4 && v <= 0));
  assert.ok(t.length >= 3);
});
