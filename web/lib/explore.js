/*
  Explore table ordering — Phase 24.

  Extracted from ExploreView so the demotion rule is unit-testable. The rule exists because
  the table is sortable and the bundle contains genuine data artifacts: sorting by beta used
  to put CHRD (beta 364, a pre/post-bankruptcy price splice) at the top of the first screen a
  visitor sees. We do not hide or alter those rows — we keep them out of the front of the
  risk orderings, in BOTH directions, and mark them.
*/

// Columns where a single out-of-band value would dominate the ordering.
export const RISK_COLS = new Set(["beta", "ann_vol"]);

export function isUnreliable(row) {
  return row.stat_quality === "unreliable";
}

/** Comparator for one column/direction. dir: 1 ascending, -1 descending. */
export function rowComparator(col, dir) {
  const demote = RISK_COLS.has(col);
  return (a, b) => {
    if (demote) {
      const ab = isUnreliable(a) ? 1 : 0;
      const bb = isUnreliable(b) ? 1 : 0;
      if (ab !== bb) return ab - bb;      // flagged rows sink, whichever way dir points
    }
    const av = a[col], bv = b[col];
    if (av == null) return 1;             // missing values sort last, always
    if (bv == null) return -1;
    if (typeof av === "number") return (av - bv) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  };
}

export function sortRows(rows, col, dir) {
  return [...rows].sort(rowComparator(col, dir));
}
