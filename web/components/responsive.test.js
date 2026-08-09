/*
  WP4's rails, as tests.

  There is no browser in this environment, so "verified on a ~390px viewport" cannot be a
  screenshot. It can, however, be a set of properties that are true of the stylesheet — and
  those are worth more than a screenshot anyway, because a screenshot passes once and a test
  keeps passing.

  What is checked here is the class of failure that actually breaks a phone: a fixed width
  wider than the viewport, a grid whose column floor cannot fit, wide content allowed to
  scroll the BODY sideways instead of itself, and a tap target too small to hit.
*/

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..");
const css = readFileSync(path.join(WEB, "app", "globals.css"), "utf8");
const ui = readFileSync(path.join(WEB, "styles", "ui.css"), "utf8");
const src = (f) => readFileSync(path.join(WEB, f), "utf8");

/** The narrowest viewport the instruction names, minus the page's own gutters. */
const VIEWPORT = 390;
const GUTTER = 14;
const CONTENT = VIEWPORT - GUTTER * 2;   // 362px of usable column

/** Split a stylesheet into { selector, body } blocks, comments stripped. */
function rules(text) {
  const clean = text.replace(/\/\*[\s\S]*?\*\//g, "");
  const out = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  while ((m = re.exec(clean))) out.push({ sel: m[1].trim(), body: m[2] });
  return out;
}

/* ------------------------------------------------------------------ widths */

test("no fixed width exceeds the phone content column without a cap", () => {
  const offenders = [];
  for (const { sel, body } of rules(css + ui)) {
    for (const m of body.matchAll(/(?<!max-|min-)width:\s*(\d+)px/g)) {
      const px = Number(m[1]);
      // A fixed width is fine as long as it is allowed to shrink.
      if (px > CONTENT && !/max-width:\s*100%/.test(body)) {
        offenders.push(`${sel} { width: ${px}px }`);
      }
    }
  }
  assert.deepEqual(offenders, [],
    `wider than a ${VIEWPORT}px viewport with no max-width escape:\n  ${offenders.join("\n  ")}`);
});

test("no grid column floor is wider than the phone content column", () => {
  // `minmax(400px, 1fr)` cannot fit, so the grid overflows its container silently.
  const offenders = [];
  for (const { sel, body } of rules(css + ui)) {
    for (const m of body.matchAll(/minmax\((\d+)px/g)) {
      if (Number(m[1]) > CONTENT) offenders.push(`${sel} → minmax(${m[1]}px …)`);
    }
  }
  assert.deepEqual(offenders, []);
});

test("the multi-column grids all collapse at the phone breakpoint", () => {
  const phone = css.slice(css.indexOf("@media (max-width: 460px)"));
  for (const sel of [".hero-stats", ".tiles", ".pie-legend"]) {
    assert.ok(phone.includes(sel),
      `${sel} is multi-column above the breakpoint and is never collapsed below it`);
  }
});

/* ------------------------------------------------------------------ overflow */

test("wide content scrolls inside its own container, never the body", () => {
  assert.match(css, /html,\s*body\s*\{[^}]*overflow-x:\s*hidden/,
    "the body must not scroll sideways on a phone");
  assert.match(css, /\.table-scroll\s*\{[^}]*overflow-x:\s*auto/,
    "the Explore table is wider than a phone and must scroll in its own container");
  assert.match(css, /\.tabs\s*\{[^}]*overflow-x:\s*auto/,
    "three tabs do not fit at 390px; they must scroll rather than wrap or overflow");
});

test("the tooltip stops floating on a phone", () => {
  // A 260px bubble centred on a term near the screen edge overflows, and pure CSS has
  // nothing to clamp a floating element against. Below the breakpoint it goes in-flow.
  const phone = css.slice(css.indexOf("@media (max-width: 460px)"));
  assert.match(phone, /\.ra-tooltip-bubble\s*\{[^}]*position:\s*static/);
});

/* ------------------------------------------------------------------ tap targets */

test("primary tap targets clear 44px", () => {
  // CSS cascades, so a selector's declarations can be spread over several rules — collect
  // every block for a selector rather than trusting the first one to carry the floor.
  const declsFor = (sel) =>
    rules(css)
      .filter((r) => r.sel.split(",").some((s) => s.trim() === sel))
      .map((r) => r.body)
      .join(";");
  assert.ok(declsFor(".chip"), ".chip has no rule at all");
  for (const sel of [".chip", ".tab", ".picker-menu li", ".explore-table th"]) {
    assert.match(declsFor(sel), /min-height:\s*44px/, `${sel} is below the 44px tap floor`);
  }
  // The kit's own controls carry the floor as a token.
  assert.match(ui, /\.ra-chip\s*\{[^}]*min-height:\s*var\(--ra-tap\)/);
});

test("the compact chip opts out explicitly, so the exemption is visible", () => {
  // .chip.tiny is deliberately under the floor — it lives in dense table rows that are not
  // primary targets. The point is that it says so rather than silently inheriting nothing.
  assert.match(css, /\.chip\.tiny\s*\{[^}]*min-height:\s*0/);
});

/* ------------------------------------------------------------------ states */

test("loading, empty and error states all exist and are used", () => {
  const states = src("components/States.js");
  for (const c of ["DonutSkeleton", "EmptyState", "ErrorState", "Skeleton"]) {
    assert.ok(states.includes(`export function ${c}`), `States.js is missing ${c}`);
  }
  // Wired, not merely written.
  assert.match(src("components/PieApp.js"), /<DonutSkeleton/, "the pie has no loading state");
  assert.match(src("components/PieApp.js"), /<ErrorState/, "a failed bundle fetch has no error state");
  assert.match(src("components/ExploreView.js"), /<EmptyState/, "Explore has no no-matches state");
  assert.match(src("components/BasketView.js"), /<EmptyState/, "the empty basket has no styled state");
});

test("the error state keeps the reader oriented instead of apologising", () => {
  const pie = src("components/PieApp.js");
  // The stale-pie trap: a failed fetch leaves the PREVIOUS beta's numbers on screen. The
  // copy has to say so, or the reader reads them as the level they just asked for.
  assert.match(pie, /still the ones for/,
    "a failed fetch must tell the reader the figures below are the previous beta's");
  assert.match(pie, /onRetry=\{retry\}/, "an error state with no way out is a dead end");
});

test("a retry actually re-runs the fetch", () => {
  // setTarget(b => b) is a no-op React bails out of, so a retry on the same beta would do
  // nothing at all. The effect needs a value that changes.
  const pie = src("components/PieApp.js");
  assert.match(pie, /setAttempt\(\(n\) => n \+ 1\)/);
  assert.match(pie, /\}, \[target, keyFor, attempt\]\)/,
    "the fetch effect does not depend on the retry counter, so Try again is inert");
});

test("the skeleton matches the shape it stands in for", () => {
  // A placeholder that resizes when data lands is just a slower flicker.
  assert.match(css, /\.skeleton-ring\s*\{[^}]*aspect-ratio:\s*1/);
  assert.match(css, /\.skeleton-ring\s*\{[^}]*mask:\s*radial-gradient/);
});

/* ------------------------------------------------------------------ legends */

test("every chart names its marks", () => {
  const panels = src("components/Panels.js");
  // Two series: a legend box, because identity must never be colour alone.
  assert.match(panels, /className="legend"/, "the growth chart has two series and no legend");
  // One series: a caption naming the ENCODING, which is what a reader of a single-series
  // chart actually needs. A swatch list would restate the title.
  assert.match(panels, /<DrawdownLegend/, "the drawdown chart never says what the shading means");
  assert.match(src("components/Charts.js"), /export function DrawdownLegend/);
  // The donut's legend is its holdings list.
  assert.match(src("components/Donut.js"), /className="pie-legend"/);
});

/* ------------------------------------------------------------------ dead code */

test("the v1 tooltip is gone, not left beside its replacement", () => {
  const panels = src("components/Panels.js");
  assert.ok(!/export function Info\b/.test(panels),
    "Panels.Info duplicates WP3's <Term>; two tooltip components is how they drift apart");
  assert.ok(!/<Info\b/.test(panels + src("components/BasketView.js") + src("components/ExploreView.js")));
});
