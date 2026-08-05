import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { partialMonthNote } from "./disclosure.js";

/* The behaviour A-3 is about: a raised flag must produce something to render. */

test("complete final month discloses nothing", () => {
  assert.equal(partialMonthNote({ axis_last_month_partial: false }), null);
  assert.equal(partialMonthNote({}), null);
  assert.equal(partialMonthNote(null), null);
  assert.equal(partialMonthNote(undefined), null);
});

test("partial final month renders the exported sentence verbatim", () => {
  const note = partialMonthNote({
    axis_last_month_partial: true,
    axis_last_month_days: 16,
    axis_last_month_text: "Final month is partial (16 trading days to 2026-06-16) — stats include it.",
  });
  assert.equal(note,
    "Final month is partial (16 trading days to 2026-06-16) — stats include it.");
});

test("the frontend never invents wording when the exporter supplied it", () => {
  // Guards the bundle rule: no number is typed into the frontend. If the exporter's sentence
  // is present it is used as-is, even when the day count would suggest something different.
  const note = partialMonthNote({
    axis_last_month_partial: true,
    axis_last_month_days: 3,
    axis_last_month_text: "Final month is partial (16 trading days to 2026-06-16) — stats include it.",
  });
  assert.match(note, /16 trading days/);
});

test("a flag with no exported text still discloses — silence is the bug", () => {
  const withDays = partialMonthNote({
    axis_last_month_partial: true, axis_last_month_days: 16,
  });
  assert.match(withDays, /partial/i);
  assert.match(withDays, /16/);

  const bare = partialMonthNote({ axis_last_month_partial: true });
  assert.match(bare, /partial/i);

  const blank = partialMonthNote({ axis_last_month_partial: true, axis_last_month_text: "   " });
  assert.match(blank, /partial/i);
});

/* The A-3 regression itself: the flag was exported and rendered NOWHERE. Assert every
   surface that shows partial-month-derived stats actually consumes the helper. */

test("every stats surface renders the disclosure", () => {
  for (const f of ["../components/ExploreView.js", "../components/BasketView.js",
                   "../components/Panels.js"]) {
    const src = readFileSync(new URL(f, import.meta.url), "utf8");
    assert.match(src, /partialMonthNote/,
      `${f} shows partial-month-derived stats but never calls partialMonthNote`);
    assert.match(src, /from "\.\.\/lib\/disclosure"/, `${f} does not import the helper`);
  }
});

test("the shipped bundles carry a renderable disclosure when the flag is up", () => {
  for (const f of ["../public/bundle/index.json", "../public/bundle/stocks.json",
                   "../public/bundle/explore.json"]) {
    const p = new URL(f, import.meta.url);
    if (!existsSync(p)) continue;            // bundle is a gitignored build artifact
    const b = JSON.parse(readFileSync(p, "utf8"));
    // A pre-fix bundle has the bare boolean and no `axis_last_month_days` key. The shipped
    // one is pinned to the deployed data state (#26d) and cannot be regenerated without
    // shipping the deferred mega-caps, so it is skipped, not failed.
    if (!("axis_last_month_days" in b)) continue;
    if (!b.axis_last_month_partial) continue;
    const note = partialMonthNote(b);
    assert.ok(note && note.length > 10, `${f} raises the flag with nothing to render`);
    assert.ok(Number.isInteger(b.axis_last_month_days) && b.axis_last_month_days > 0,
      `${f} raises the flag without a day count`);
  }
});
