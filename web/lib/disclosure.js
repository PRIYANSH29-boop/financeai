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
  if (!bundle || !bundle.axis_last_month_partial) return null;
  const text = bundle.axis_last_month_text;
  if (typeof text === "string" && text.trim()) return text;
  // Flag raised without the exported sentence: the export validator refuses this, so it can
  // only mean a hand-edited or stale bundle. Disclose anyway — silence is the failure mode
  // A-3 was about, and a vaguer sentence beats none.
  const days = bundle.axis_last_month_days;
  return Number.isFinite(days)
    ? `Final month is partial (${days} trading days) — stats include it.`
    : "Final month is partial — stats include it.";
}
