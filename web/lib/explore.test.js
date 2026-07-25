/*
  node:test suite for the Explore table's #24 ordering rule (lib/explore.js).

  The defect this guards: the table is sortable, and the wide universe contains genuine price
  artifacts (CHRD's beta of 364 comes from splicing pre- and post-bankruptcy Whiting prices
  into one adjusted-close series). Clicking "Beta" put that at the top of the first screen.
  The rule sinks flagged rows in the risk orderings without altering or hiding them — and the
  last test runs it against the SHIPPED bundle, because a rule that passes on fixtures while
  the live page still leads with garbage has fixed nothing.
*/

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { sortRows, rowComparator, RISK_COLS, isUnreliable } from "./explore.js";

const ok = (ticker, beta, ann_vol) => ({ ticker, beta, ann_vol, stat_quality: "ok" });
const bad = (ticker, beta, ann_vol) =>
  ({ ticker, beta, ann_vol, stat_quality: "unreliable" });

const ROWS = [ok("AAA", 1.1, 0.25), bad("JUNK", 364.5, 1.26), ok("BBB", 1.8, 0.4),
              ok("CCC", 0.3, 0.15), bad("SHAKY", 6.0, 1.6)];

test("sorting by beta descending does not lead with a flagged row", () => {
  const got = sortRows(ROWS, "beta", -1).map((r) => r.ticker);
  assert.deepEqual(got, ["BBB", "AAA", "CCC", "JUNK", "SHAKY"]);
});

test("sorting by beta ascending also sinks flagged rows, not just descending", () => {
  const got = sortRows(ROWS, "beta", 1).map((r) => r.ticker);
  assert.deepEqual(got, ["CCC", "AAA", "BBB", "SHAKY", "JUNK"]);
});

test("flagged rows keep their relative order at the bottom", () => {
  const tail = sortRows(ROWS, "ann_vol", -1).slice(-2).map((r) => r.ticker);
  assert.deepEqual(tail, ["SHAKY", "JUNK"]);   // 1.6 then 1.26, still sorted among themselves
});

test("flagged rows are demoted, never dropped", () => {
  assert.equal(sortRows(ROWS, "beta", -1).length, ROWS.length);
  assert.equal(sortRows(ROWS, "ann_vol", 1).length, ROWS.length);
});

test("non-risk columns are NOT demoted — a ticker sort stays alphabetical", () => {
  const got = sortRows(ROWS, "ticker", 1).map((r) => r.ticker);
  assert.deepEqual(got, ["AAA", "BBB", "CCC", "JUNK", "SHAKY"]);
});

test("only beta and ann_vol are risk columns", () => {
  assert.deepEqual([...RISK_COLS].sort(), ["ann_vol", "beta"]);
  assert.ok(!RISK_COLS.has("last_return"));
});

test("null values sort last regardless of direction", () => {
  const rows = [ok("AAA", 1.0, 0.2), ok("NUL", null, null), ok("BBB", 2.0, 0.3)];
  assert.equal(sortRows(rows, "beta", -1).at(-1).ticker, "NUL");
  assert.equal(sortRows(rows, "beta", 1).at(-1).ticker, "NUL");
});

test("isUnreliable reads the exporter's stat_quality field", () => {
  assert.ok(isUnreliable({ stat_quality: "unreliable" }));
  assert.ok(!isUnreliable({ stat_quality: "ok" }));
  assert.ok(!isUnreliable({}));
});

test("comparator is a pure function of the two rows", () => {
  const cmp = rowComparator("beta", -1);
  assert.equal(cmp(ok("A", 2, 0.1), ok("B", 1, 0.1)), -1);
  assert.equal(cmp(bad("A", 500, 2), ok("B", 1, 0.1)), 1);
});

// ---- against the real shipped bundle
test("shipped bundle: no flagged row appears in the first screen of a beta sort", () => {
  let explore;
  try {
    explore = JSON.parse(readFileSync("public/bundle/explore.json", "utf8"));
  } catch {
    return;   // bundle not built in this checkout
  }
  const flagged = explore.rows.filter(isUnreliable);
  assert.ok(flagged.length > 0, "expected the wide universe to contain flagged rows");

  for (const dir of [-1, 1]) {
    const firstScreen = sortRows(explore.rows, "beta", dir).slice(0, 100);
    const leaked = firstScreen.filter(isUnreliable).map((r) => r.ticker);
    assert.deepEqual(leaked, [], `flagged rows in the first 100 of beta dir=${dir}`);
  }
  // and the specific offender the reviewer named is genuinely in the flagged set
  const chrd = explore.rows.find((r) => r.ticker === "CHRD");
  if (chrd) {
    assert.ok(isUnreliable(chrd), "CHRD (beta 364, bankruptcy splice) is not flagged");
    assert.ok(!chrd.scored, "CHRD must not be basket-eligible");
  }
});

test("shipped bundle: every flagged row carries a human-readable reason", () => {
  let explore;
  try {
    explore = JSON.parse(readFileSync("public/bundle/explore.json", "utf8"));
  } catch {
    return;
  }
  for (const r of explore.rows.filter(isUnreliable)) {
    assert.ok(r.stat_note, `${r.ticker} flagged with no stat_note for the tooltip`);
    assert.ok(r.stat_flags.length > 0, `${r.ticker} flagged with no flag codes`);
  }
});
