// Formatting helpers. Every number the UI shows passes through here, so precision and
// sign conventions are decided once rather than per component.

export const money = (v, currency = "GBP") =>
  new Intl.NumberFormat("en-GB", {
    style: "currency", currency, maximumFractionDigits: 0,
  }).format(v ?? 0);

export const pct = (v, dp = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(dp)}%`;

export const num = (v, dp = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(dp);

// Drift deltas are percentage POINTS of weight, and always carry an explicit sign so a
// small positive value cannot be misread as a level.
export const pp = (v, dp = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(dp)}pp`;

export const beta = (v) => (v === null || v === undefined ? "—" : v.toFixed(2));
