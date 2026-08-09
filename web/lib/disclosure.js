/*
  Partial-final-month disclosure — Phase 28, fixing #25 finding A-3.

  The bundles are built by resampling daily prices to month-end with `.last()`. When the
  panel stops mid-month the final bucket is LABELLED at month-end but built from only the
  days that exist, and every stat computed off that axis — annualised vol, beta, Sharpe,
  max drawdown, last-month return, the movers cards, every basket scorecard — silently
  includes it.

  #24 exported `axis_last_month_partial` and nothing rendered it: true, machine-readable,
  and invisible to the user for a full release. Extracted here, like the explore.js ordering
  rule, so the decision to show it is unit-testable and lives in exactly one place. The
  exporter ships the sentence itself (`axis_last_month_text`) — this file never composes
  wording or recomputes a number, it only decides whether there is something to say.
*/

/**
 * The disclosure line for a bundle (index.json / stocks.json / explore.json), or null when
 * the final month is complete and there is nothing to disclose.
 *
 * Returns null rather than a placeholder so callers render with `{note && ...}` and a
 * complete-month bundle produces no empty element.
 */
export function partialMonthNote(bundle) {
  if (!bundle) return null;
  // #30-B added a third outcome. A DROPPED final bucket leaves the stats axis clean, so
  // `axis_last_month_partial` is correctly false — but the bundle's as-of date still
  // advertises data more recent than anything the figures were computed from, and saying
  // nothing about that gap is A-3 pointed the other way.
  const dropped = bundle.axis_last_month_action === "dropped";
  if (!bundle.axis_last_month_partial && !dropped) return null;
  const text = bundle.axis_last_month_text;
  if (typeof text === "string" && text.trim()) return text;
  // Flag raised without the exported sentence: the export validator refuses this, so it can
  // only mean a hand-edited or stale bundle. Disclose anyway — silence is the failure mode
  // A-3 was about, and a vaguer sentence beats none.
  const days = bundle.axis_last_month_days;
  if (dropped) {
    return Number.isFinite(days)
      ? `Final month held only ${days} trading days — excluded from every statistic here.`
      : "Final month was too short to annualise — excluded from every statistic here.";
  }
  return Number.isFinite(days)
    ? `Final month is partial (${days} trading days) — stats include it.`
    : "Final month is partial — stats include it.";
}

/*
  #32 WP3 — the trust chips.

  The stacked red caveat boxes said the right things in the most alarming possible way: a
  wall of warnings reads as boilerplate and gets skipped, which is the opposite of
  disclosure. The same sentences now render as calm one-line chips that expand.

  THE RULE THIS FUNCTION EXISTS TO ENFORCE: presentation may change, wording may not. Every
  chip's visible line is the exporter's string, byte for byte — this file never rewrites,
  shortens or softens one, and never drops one. The optional `detail` is ADDITIVE: plain
  English about why the caveat matters, from lib/glossary.js. glossary.test.js asserts the
  verbatim property against the shipped bundle, so a future edit that paraphrases a
  disclosure fails the suite rather than shipping.
*/

import { caveatDetail, PARTIAL_MONTH_DETAIL, DROPPED_MONTH_DETAIL } from "./glossary.js";

/**
 * Every disclosure for a bundle, in the order they should be shown.
 *
 * Takes the caveat-carrying bundle (index.json) and optionally the bundle whose stats are
 * on screen (stocks.json / explore.json), because the partial-month note belongs to the
 * axis being displayed, not to whichever file happened to carry the caveat list.
 */
export function trustChips(bundle, statsBundle = bundle) {
  const chips = [];

  const partial = partialMonthNote(statsBundle);
  if (partial) {
    // First, because it is the one that is true *today* rather than always. The detail
    // must follow the ruling: telling a reader that stats "include it" when the bucket was
    // dropped would be a wrong explanation attached to a correct sentence.
    const dropped = statsBundle?.axis_last_month_action === "dropped";
    chips.push({
      id: "partial-month",
      text: partial,
      detail: dropped ? DROPPED_MONTH_DETAIL : PARTIAL_MONTH_DETAIL,
      transient: true,
    });
  }

  for (const text of bundle?.caveats ?? []) {
    if (typeof text !== "string" || !text.trim()) continue;
    chips.push({ id: text, text, detail: caveatDetail(text), transient: false });
  }

  return chips;
}
