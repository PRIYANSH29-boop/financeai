/*
  Design-token integrity.

  Two failure modes this catches, both of which have shipped in real design systems:

    1. A component references `var(--ra-suface-2)` and silently renders transparent,
       because CSS has no such thing as an undefined-variable error.
    2. Someone "brightens" a token and quietly drops a caption below readable contrast.

  Both are cheap to check from the CSS text itself, so they are checked on every run
  rather than trusted to a screenshot.
*/

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "..");
const tokensCss = readFileSync(path.join(WEB, "styles", "tokens.css"), "utf8");
const uiCss = readFileSync(path.join(WEB, "styles", "ui.css"), "utf8");

/** Token names DEFINED in tokens.css (`--ra-foo: value`). */
const defined = new Set([...tokensCss.matchAll(/^\s*(--ra-[a-z0-9-]+)\s*:/gim)].map((m) => m[1]));
/**
 * Token names REFERENCED anywhere (`var(--ra-foo)` / `var(--ra-foo, fallback)`).
 *
 * The trailing `[),]` is what makes this usable on the JS files too: a template literal
 * like `var(--ra-text-${size})` is a name computed at runtime, so it does not match and
 * is not reported as an undefined token. Those live in the demo gallery and in Num, and
 * their suffixes come from the scale lists in the same file.
 */
const referenced = (css) =>
  new Set([...css.matchAll(/var\(\s*(--ra-[a-z0-9-]+?)\s*[),]/g)].map((m) => m[1]));

/** Hex value of a token, following one level of aliasing. */
function tokenHex(name) {
  const m = tokensCss.match(new RegExp(`${name}\\s*:\\s*(#[0-9a-f]{6})`, "i"));
  return m ? m[1] : null;
}

/** WCAG relative luminance / contrast ratio. */
const lum = (hex) => {
  const c = [1, 3, 5].map((i) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const contrast = (a, b) => {
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

/* ------------------------------------------------------------------ integrity */

test("tokens.css defines a non-trivial token set", () => {
  assert.ok(defined.size > 50, `only ${defined.size} tokens defined`);
});

test("every token ui.css uses is defined in tokens.css", () => {
  const missing = [...referenced(uiCss)].filter((t) => !defined.has(t));
  assert.deepEqual(missing, [], `ui.css references undefined tokens: ${missing.join(", ")}`);
});

test("every token tokens.css uses is defined in tokens.css", () => {
  const missing = [...referenced(tokensCss)].filter((t) => !defined.has(t));
  assert.deepEqual(missing, [], `tokens.css references undefined tokens: ${missing.join(", ")}`);
});

test("every token the ui components use is defined", () => {
  // Inline styles like style={{ fontSize: "var(--ra-text-lg)" }} bypass the stylesheet
  // entirely, so they get the same check.
  const missing = new Set();
  for (const f of readdirSync(HERE).filter((f) => f.endsWith(".js") && !f.endsWith(".test.js"))) {
    for (const t of referenced(readFileSync(path.join(HERE, f), "utf8"))) {
      if (!defined.has(t)) missing.add(`${f}:${t}`);
    }
  }
  assert.deepEqual([...missing], []);
});

test("ui.css contains no raw colour literals — everything goes through a token", () => {
  // A hex in ui.css is a token that escaped tokens.css, which is how a design system
  // stops being re-themeable.
  const hexes = uiCss.match(/#[0-9a-fA-F]{3,8}\b/g) || [];
  assert.deepEqual(hexes, []);
});

test("the v3 tokens cannot collide with the v1 light tokens", () => {
  // v1 owns :root and unprefixed names (--accent, --bg, …). v3 owns .ra-root and --ra-*.
  // If either half of that ever stops being true, the live pages restyle themselves.
  assert.ok(!/^\s*:root\s*\{/m.test(tokensCss), "tokens.css must not define on :root");
  for (const t of defined) assert.ok(t.startsWith("--ra-"), `${t} is not namespaced`);
  const stray = [...uiCss.matchAll(/var\(\s*(--(?!ra-)[a-z0-9-]+)/g)].map((m) => m[1]);
  assert.deepEqual(stray, [], `ui.css reaches into v1 tokens: ${stray.join(", ")}`);
});

test("every ui.css rule is scoped under .ra-root", () => {
  // Selectors only — comments and declaration bodies are stripped first.
  const selectors = uiCss
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\{[^{}]*\}/g, "{}")
    .split("}")
    .map((chunk) => chunk.split("{")[0].trim())
    .filter((s) => s && !s.startsWith("@"));
  const unscoped = selectors.flatMap((s) => s.split(",").map((x) => x.trim()))
    .filter((s) => s && !s.startsWith(".ra-root"));
  assert.deepEqual(unscoped, [], `unscoped selectors would leak into v1: ${unscoped.join(" | ")}`);
});

/* ------------------------------------------------------------------ contrast */

const SURFACES = {
  "--ra-bg": null,
  "--ra-surface": null,
  "--ra-surface-2": null,
};

test("body ink clears WCAG AA on every surface", () => {
  for (const s of Object.keys(SURFACES)) {
    for (const ink of ["--ra-text", "--ra-text-muted", "--ra-text-faint", "--ra-num-neutral"]) {
      const r = contrast(tokenHex(ink), tokenHex(s));
      assert.ok(r >= 4.5, `${ink} on ${s} is ${r.toFixed(2)}:1, below AA 4.5`);
    }
  }
});

test("the accent and the loss colour are readable as text, not just as fills", () => {
  // Both are used for figures (a gain, a drawdown), so they answer to the text bar.
  for (const s of Object.keys(SURFACES)) {
    for (const ink of ["--ra-accent", "--ra-accent-strong", "--ra-loss"]) {
      const r = contrast(tokenHex(ink), tokenHex(s));
      assert.ok(r >= 4.5, `${ink} on ${s} is ${r.toFixed(2)}:1, below AA 4.5`);
    }
  }
});

test("reserved chart marks clear the 3:1 non-text bar", () => {
  for (const mark of ["--ra-cash", "--ra-benchmark", "--ra-accent-dim"]) {
    const r = contrast(tokenHex(mark), tokenHex("--ra-surface"));
    assert.ok(r >= 3, `${mark} on surface is ${r.toFixed(2)}:1, below 3:1`);
  }
});

test("the surface ladder is ordered and separable", () => {
  const ladder = ["--ra-bg", "--ra-surface", "--ra-surface-2", "--ra-surface-3"].map(tokenHex);
  for (let i = 1; i < ladder.length; i++) {
    assert.ok(lum(ladder[i]) > lum(ladder[i - 1]),
              `surface step ${i} is not lighter than the one below it`);
  }
});

/* ------------------------------------------------------------------ system rules */

test("motion is switched off for prefers-reduced-motion", () => {
  assert.match(tokensCss, /@media\s*\(prefers-reduced-motion:\s*reduce\)/);
  assert.match(tokensCss, /animation-duration:\s*1ms\s*!important/);
});

test("the tap-target floor is at least 44px and the chip honours it", () => {
  const tap = tokensCss.match(/--ra-tap:\s*(\d+)px/);
  assert.ok(tap && Number(tap[1]) >= 44, "--ra-tap must be >= 44px");
  assert.match(uiCss, /\.ra-chip\s*\{[^}]*min-height:\s*var\(--ra-tap\)/);
});

test("no focus outline is removed anywhere in the kit", () => {
  assert.doesNotMatch(uiCss, /outline:\s*(none|0)\b/);
});
