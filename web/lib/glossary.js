/*
  The glossary — ONE definition per term, for the whole product.

  #32 WP3. The cold read was "jargon walls": the UI says cash sleeve, Sharpe, Sortino,
  beta, drift, survivorship, Rank IC at a reader who has been given no way to find out what
  any of them mean. Definitions used to be written inline at each call site, which is how
  the same word ends up explained two different ways on two tabs.

  Rules for anything added here:

    * PLAIN ENGLISH FIRST, then the precision. A definition that needs a second definition
      to parse has not done its job.
    * NEVER FORWARD-LOOKING. No definition may describe what a number predicts, promises or
      is expected to do — every figure in this product is a historical characterisation of
      today's weights. glossary.test.js enforces this with a banned-phrase list.
    * NO NUMBERS. A definition explains a term; the bundle supplies the values. If a
      definition contained a figure it would be a number typed into the frontend, which is
      the one thing this codebase does not do.

  Keys are lowercase and matched exactly by `define()`.
*/

export const GLOSSARY = {
  beta:
    "How much your pie moves when the market moves. Beta 0.5 means roughly half the " +
    "market's swing, up and down. Beta 1 means it moves with the market.",

  "target beta":
    "The risk level you asked for. The engine holds cash to hit it — it never borrows to " +
    "reach a higher one.",

  "realised beta":
    "The beta the pie actually had over the measured history, as opposed to the one it " +
    "was aiming for. The two differ because a portfolio is built from whole positions, " +
    "not from a dial.",

  "cash sleeve":
    "The share of your money held as cash rather than stock. It is the control, not a " +
    "leftover: cash has a beta of zero, so holding more of it pulls the whole pie's risk " +
    "down. A low-risk pie is mostly cash, and this product says so rather than hiding it.",

  sharpe:
    "Return per unit of risk, measured over the committed history. Higher means the " +
    "return came with a smoother ride. It treats an upward jump and a downward one as " +
    "equally 'risky', which is why Sortino sits next to it.",

  sortino:
    "Like Sharpe, but it only counts the downward moves as risk. Rising fast is not the " +
    "kind of volatility anyone wants protection from.",

  drawdown:
    "How far the pie fell from its own previous peak. It is the number that decides " +
    "whether a strategy is actually holdable, because it is the loss you would have had " +
    "to sit through.",

  "max drawdown":
    "The worst peak-to-trough fall over the measured history — the deepest hole the pie " +
    "climbed out of.",

  volatility:
    "How much the value bounces around from month to month. Not the same thing as loss: " +
    "a volatile pie can end higher, it just gets there less comfortably.",

  drift:
    "What happens to your weights after you buy. Prices move, so winners grow into a " +
    "larger share of the pie and losers shrink — by the next rebalance you no longer hold " +
    "the mix you chose. Drift is measured in percentage points of weight, and it is " +
    "information rather than a gain or a loss.",

  rebalance:
    "Selling a little of what grew and topping up what shrank, to put the pie back on the " +
    "weights it was meant to have.",

  // Spelled out rather than written with numerals: an illustrative percentage inside a
  // definition is exactly the shape a reader can mistake for one of this page's results.
  "percentage points":
    "The gap between two percentages. A weight going from four percent to five percent " +
    "moved one percentage point — not twenty-five percent. Written 'pp' so it cannot be " +
    "misread as a return.",

  survivorship:
    "The universe is today's index members applied to the whole history, so companies " +
    "that failed or were removed are missing from it. That flatters every backtest, " +
    "including this one. The real fix is a point-in-time membership list, which this " +
    "project does not have.",

  "rank ic":
    "How well the model's ordering of stocks matched what actually happened next — " +
    "ranking skill across the whole list, rather than the payoff of the extremes. It is " +
    "the measure that disagrees most usefully with Sharpe.",

  benchmark:
    "The comparison the pie is measured against. A number is not good or bad on its own; " +
    "it is good or bad next to what you could have had instead.",

  "walk-forward":
    "Training only on the past and testing on the period that followed, then rolling " +
    "forward. It is how you find out whether a model works, rather than whether it can " +
    "memorise.",

  embargo:
    "A gap left between the training data and the test data, so a label that overlaps the " +
    "test window cannot leak the answer backwards into training.",

  "point-in-time":
    "Using the value that was actually published at the time, rather than today's " +
    "restated version. Without it, a model gets to see figures nobody had yet.",

  simulation:
    "No money moves. This is a historical characterisation of a set of weights, built to " +
    "be inspected — not a product you can buy and not advice.",

  turnover:
    "How much of the pie is bought and sold at each rebalance. It matters because every " +
    "trade costs something, and a strategy that looks good before costs can lose after " +
    "them.",

  "frozen model":
    "The ranker was fitted once and is never refitted to make later numbers look better. " +
    "A model retuned after seeing its own results is not being tested any more.",
};

/**
 * The definition for a term, or null when there isn't one.
 *
 * Returns null rather than a placeholder so a caller renders with `{def && …}` and an
 * unknown term degrades to plain text instead of an empty tooltip.
 */
export function define(term) {
  if (typeof term !== "string") return null;
  return GLOSSARY[term.trim().toLowerCase()] ?? null;
}

/*
  What each shipped caveat MEANS, for the "tap to expand" layer.

  These are keyed by the caveat's own text so the chip's visible line stays the exporter's
  wording, byte for byte — the detail is additive. A caveat with no entry here still
  renders; it simply has nothing to expand, which is a missing explanation rather than a
  missing disclosure.
*/
export const CAVEAT_DETAIL = {
  "23-month simulated track — too short to be statistically significant":
    "Twenty-three months is roughly two dozen independent observations. A handful of good " +
    "or bad months moves every figure on this page, so treat the direction as interesting " +
    "and the precision as noise. It is a live, growing record, not a verdict.",

  "survivorship-biased universe":
    "The stock list is today's index members applied backwards through history, so every " +
    "company that failed or dropped out is missing. That makes the past look kinder than " +
    "it was — to this strategy and to its benchmark alike.",

  "target beta estimated from calm markets; real beta rises in crashes":
    "Beta is measured across the whole history, most of which is uneventful. In a real " +
    "fall, correlations rise and holdings move together more than the headline number " +
    "suggests. The realised beta inside the worst drawdown window is shown next to the " +
    "headline for exactly this reason.",

  "educational simulation, not investment advice":
    "Nothing here is a recommendation and no money moves. The point of the product is to " +
    "show its working — where the numbers come from, and where they stop being " +
    "trustworthy.",
};

/**
 * The partial-month note is generated per bundle (it carries a day count), so it cannot be
 * keyed by text like the fixed caveats. Its explanation is constant, which is why it lives
 * on its own rather than in the map above.
 */
export const PARTIAL_MONTH_DETAIL =
  "Monthly figures are built by taking the last price in each month. When the data stops " +
  "mid-month the final bucket is labelled at month-end but built from only the days that " +
  "exist, and every statistic on the page — volatility, beta, Sharpe, drawdown — includes " +
  "it. A short final month is noisier than a full one.";

/** The expansion for a caveat line, or null when none is written for it. */
export function caveatDetail(text) {
  if (typeof text !== "string") return null;
  return CAVEAT_DETAIL[text.trim()] ?? null;
}
