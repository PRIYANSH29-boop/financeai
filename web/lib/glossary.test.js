/*
  WP3's rails, as tests.

  The instruction is explicit: "disclosures may be RESTYLED but never removed or weakened",
  and "wording of disclosures unchanged — only their presentation". Both are the sort of
  promise that quietly stops being true two refactors later, so both are asserted here
  against the SHIPPED bundle rather than a fixture.

  The verbatim test is the important one. It fails if anyone paraphrases, truncates or
  softens a caveat on the way to the screen.
*/

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { GLOSSARY, CAVEAT_DETAIL, define, caveatDetail } from "./glossary.js";
import { trustChips, partialMonthNote } from "./disclosure.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const bundle = (f) =>
  JSON.parse(readFileSync(path.join(HERE, "..", "public", "bundle", f), "utf8"));
const index = bundle("index.json");

/* ------------------------------------------------------- the disclosure rails */

test("every shipped caveat reaches the screen, byte for byte", () => {
  const shown = trustChips(index).map((c) => c.text);
  for (const caveat of index.caveats) {
    assert.ok(shown.includes(caveat),
              `caveat is missing or was reworded on the way to the UI: ${caveat}`);
  }
});

test("no caveat is dropped, and none is invented", () => {
  const chips = trustChips(index);
  const fromCaveats = chips.filter((c) => !c.transient);
  assert.equal(fromCaveats.length, index.caveats.length);
  assert.deepEqual(fromCaveats.map((c) => c.text), index.caveats);
});

test("the detail layer is additive — it never replaces the caveat's own words", () => {
  for (const chip of trustChips(index)) {
    if (!chip.detail) continue;
    assert.notEqual(chip.detail, chip.text);
    // the chip's visible line is still the exporter's string
    assert.ok(chip.text.length > 0);
  }
});

test("every shipped caveat has an explanation written for it", () => {
  // Not a hard rail — a chip with no detail still discloses — but a missing entry means a
  // reader gets the jargon and not the meaning, which is the failure WP3 exists to fix.
  const missing = index.caveats.filter((c) => !caveatDetail(c));
  assert.deepEqual(missing, [], `no plain-English detail for: ${missing.join(" | ")}`);
});

test("the partial-month note leads the chips when the bundle raises the flag", () => {
  const flagged = {
    ...index,
    axis_last_month_partial: true,
    axis_last_month_text: "Final month is partial (3 trading days) — stats include it.",
  };
  const chips = trustChips(flagged);
  assert.equal(chips[0].id, "partial-month");
  assert.equal(chips[0].text, flagged.axis_last_month_text);
  assert.equal(chips.length, index.caveats.length + 1);
});

test("a complete final month adds no chip — silence is correct there", () => {
  const clean = { ...index, axis_last_month_partial: false };
  assert.equal(partialMonthNote(clean), null);
  assert.equal(trustChips(clean).length, index.caveats.length);
});

test("the partial-month note follows the bundle whose stats are on screen", () => {
  // Explore's table and the pie's scorecard are built off different axes; the note belongs
  // to the one being displayed, not to whichever file carried the caveat list.
  const stats = {
    axis_last_month_partial: true,
    axis_last_month_text: "Final month is partial (5 trading days) — stats include it.",
  };
  const chips = trustChips(index, stats);
  assert.equal(chips[0].text, stats.axis_last_month_text);
});

test("trustChips survives a bundle with no caveats rather than throwing", () => {
  assert.deepEqual(trustChips({}), []);
  assert.deepEqual(trustChips(null), []);
  assert.deepEqual(trustChips({ caveats: ["", "  ", "real one"] }).map((c) => c.text),
                   ["real one"]);
});

/* ------------------------------------------------------- the glossary itself */

test("every glossary entry has a real definition", () => {
  for (const [term, def] of Object.entries(GLOSSARY)) {
    assert.equal(typeof def, "string");
    assert.ok(def.trim().length > 40, `${term} is too thin to be useful`);
  }
});

test("no definition is forward-looking", () => {
  // The product's central invariant on the frontend: every figure is a historical
  // characterisation of today's weights, never a forecast. A definition that says a number
  // predicts, promises or is expected to do something breaks that in the place a reader is
  // most likely to trust it.
  const banned = [
    /\bwill (?:return|be worth|grow|rise|fall|make)\b/i,
    /\bexpected return\b/i,
    /\bguarantee/i,
    /\bpredicts? (?:your|the) (?:return|profit)/i,
    /\bshould (?:return|earn|make)\b/i,
    /\bforecasts?\b/i,
  ];
  for (const [term, def] of Object.entries({ ...GLOSSARY, ...CAVEAT_DETAIL })) {
    for (const re of banned) {
      assert.ok(!re.test(def), `"${term}" reads as forward-looking: ${re}`);
    }
  }
});

test("no definition contains a figure — the bundle supplies values, not this file", () => {
  // "Beta 0.5" and "beta of zero" are naming a scale point, not quoting a result, so the
  // check targets the shapes a real number takes: percentages, money, and decimals.
  const looksLikeData = /\d+(?:\.\d+)?%|[£$]\s?\d|\b\d+\.\d{2,}\b/;
  for (const [term, def] of Object.entries({ ...GLOSSARY, ...CAVEAT_DETAIL })) {
    assert.ok(!looksLikeData.test(def), `"${term}" has a number typed into it: ${def}`);
  }
});

test("define is case- and space-insensitive, and misses cleanly", () => {
  assert.equal(define("Sharpe"), GLOSSARY.sharpe);
  assert.equal(define("  CASH SLEEVE "), GLOSSARY["cash sleeve"]);
  assert.equal(define("not a term"), null);
  assert.equal(define(undefined), null);
  assert.equal(define(42), null);
});

test("the terms the UI actually renders all resolve", () => {
  // Guards the <Term> call sites: a typo'd key degrades to plain text, which is silent.
  for (const k of ["sharpe", "max drawdown", "realised beta", "target beta",
                   "cash sleeve", "survivorship", "rank ic", "drift"]) {
    assert.ok(define(k), `UI renders <Term> for "${k}" but the glossary has no entry`);
  }
});

test("caveatDetail misses cleanly rather than throwing", () => {
  assert.equal(caveatDetail("something never shipped"), null);
  assert.equal(caveatDetail(null), null);
});
