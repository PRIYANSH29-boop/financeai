"""
Cross-sectional feature builder — RankAlpha Phase 2 (FEATURES).

Reads the Phase 1 panel (`data/sp500_panel.parquet`) and produces a leakage-checked
feature table (`data/sp500_features.parquet`) for cross-sectional ranking.

This is Phase 2 only — NO labels here (that's Phase 3).

Pipeline:
  A. Sanity gate          — abort loudly on bad data (prices, nulls, duplicates).
  B. Min-history filter    — point-in-time eligibility: >=253 trading days on/before t.
  C. Raw features          — momentum / reversal / vol / liquidity / size, per stock.
  D. Cross-sectional rank  — within each date, percentile-rank across eligible stocks.

Leakage guarantees (see module-level checks):
  * Every feature uses only data <= t (groupby('ticker') shifts/rolling — never peeks
    forward, never bleeds across tickers).
  * Cross-sectional normalization uses ONLY that day's eligible cross-section
    (groupby('date').rank) — no full-period statistics, no global scaler fit.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sp500_features")

# Eligibility threshold. The longest feature, mom_12_1m, references adj_close[t-252],
# which only exists once a stock has 253 observations (252-day lookback + the current
# day). Requiring >=253 makes "eligible" == "all features computable" — no NaN rows.
MIN_HISTORY = 253

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]

# Raw feature columns, in output order. Each gets a matching `<name>_rank`.
FEATURE_COLS = [
    "mom_3m",
    "mom_6m",
    "mom_12_1m",
    "reversal_1m",
    "vol_6m",
    "liquidity",
    "size",
]


# --------------------------------------------------------------------- A. sanity
def sanity_gate(df: pd.DataFrame) -> None:
    """Run the three Phase-2 data-quality checks. Print pass/fail; abort on any fail."""
    print("\n" + "=" * 60)
    print("DATA SANITY GATE")
    print("=" * 60)

    results = {}

    # 1. No zero or negative prices in any OHLC / adj_close column.
    nonpos = (df[PRICE_COLS] <= 0).sum()
    nonpos_total = int(nonpos.sum())
    results["no_nonpositive_prices"] = nonpos_total == 0
    print(f"[{_mark(results['no_nonpositive_prices'])}] No zero/negative prices "
          f"(found {nonpos_total})")
    if nonpos_total:
        for col, n in nonpos[nonpos > 0].items():
            print(f"        {col}: {int(n)} bad rows")

    # 2. No nulls in adj_close.
    n_null_adj = int(df["adj_close"].isna().sum())
    results["no_null_adj_close"] = n_null_adj == 0
    print(f"[{_mark(results['no_null_adj_close'])}] No nulls in adj_close "
          f"(found {n_null_adj})")

    # 3. Zero duplicate (date, ticker) rows.
    n_dupes = int(df.duplicated(subset=["date", "ticker"]).sum())
    results["no_duplicate_keys"] = n_dupes == 0
    print(f"[{_mark(results['no_duplicate_keys'])}] No duplicate (date, ticker) rows "
          f"(found {n_dupes})")

    print("=" * 60)
    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        logger.error("Sanity gate FAILED: %s — aborting.", ", ".join(failed))
        sys.exit(1)
    print("Sanity gate PASSED.\n")


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------- B + C. eligibility/feats
def build_raw_features(df: pd.DataFrame):
    """Add eligibility flag and raw feature columns. Returns (df, stats)."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False)
    ac = df["adj_close"]

    # Past prices (grouped shift never crosses tickers and only looks backward).
    ac_21 = g["adj_close"].shift(21)
    ac_63 = g["adj_close"].shift(63)
    ac_126 = g["adj_close"].shift(126)
    ac_252 = g["adj_close"].shift(252)

    df["mom_3m"] = ac / ac_63 - 1
    df["mom_6m"] = ac / ac_126 - 1
    # 12m-minus-1m: return from t-252 to t-21 (skip the most recent month).
    df["mom_12_1m"] = ac_21 / ac_252 - 1
    df["reversal_1m"] = ac / ac_21 - 1

    # Daily simple returns (grouped, so the first obs per ticker is NaN, not a jump).
    df["ret_1d"] = g["adj_close"].pct_change()
    df["vol_6m"] = (
        df.groupby("ticker", sort=False)["ret_1d"]
        .transform(lambda s: s.rolling(126).std())
    )

    vol_avg_20 = g["volume"].transform(lambda s: s.rolling(20).mean())
    df["liquidity"] = df["volume"] / vol_avg_20

    df["size"] = np.log(ac)

    # B. Point-in-time eligibility: >=253 trading days of history on/before t.
    # cumcount() is 0-based, so position-on/before-t = cumcount()+1; eligible when >=253,
    # i.e. adj_close[t-252] exists and every feature is computable (no NaN at the boundary).
    obs_on_or_before_t = g.cumcount() + 1
    df["eligible"] = obs_on_or_before_t >= MIN_HISTORY

    stats = {"total_stock_days": len(df)}
    stats["removed_min_history"] = int((~df["eligible"]).sum())
    return df, stats


# ------------------------------------------------------ D. cross-sectional rank
def cross_sectional_rank(elig: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank each feature within each date across that day's eligible stocks."""
    for c in FEATURE_COLS:
        # rank(pct=True) → percentile in (0, 1], computed per date only.
        elig[c + "_rank"] = elig.groupby("date")[c].rank(pct=True)
    return elig


def build_features(panel_path="data/sp500_panel.parquet",
                   out_path="data/sp500_features.parquet"):
    panel_path = Path(panel_path)
    logger.info("Loading panel from %s", panel_path)
    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Panel: %d rows, %d tickers, %s -> %s",
                len(df), df["ticker"].nunique(),
                df["date"].min().date(), df["date"].max().date())

    # A. Sanity gate (aborts on failure).
    sanity_gate(df)

    # B + C. Eligibility + raw features.
    df, stats = build_raw_features(df)

    # Keep only eligible stock-days.
    elig = df[df["eligible"]].copy()

    # Safety net: with MIN_HISTORY=253 every eligible row is fully computable, so this
    # should drop zero rows. Kept as a defensive guard against any unexpected NaN.
    before = len(elig)
    elig = elig.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    stats["removed_boundary_nan"] = before - len(elig)
    stats["final_rows"] = len(elig)

    # D. Cross-sectional rank-normalization.
    elig = cross_sectional_rank(elig)

    rank_cols = [c + "_rank" for c in FEATURE_COLS]
    out_cols = ["date", "ticker"] + FEATURE_COLS + rank_cols
    out = elig[out_cols].sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path = Path(out_path)
    out.to_parquet(out_path, index=False)
    logger.info("Saved features -> %s (%d rows, %d cols)",
                out_path, len(out), len(out.columns))

    print_report(out, stats)
    return out, stats


def print_report(out: pd.DataFrame, stats: dict):
    print("=" * 60)
    print("FEATURE BUILD SUMMARY — Phase 2")
    print("=" * 60)
    print(f"Total stock-days (panel)      : {stats['total_stock_days']:,}")
    print(f"Removed — min-history (<{MIN_HISTORY})  : {stats['removed_min_history']:,}")
    print(f"Removed — 12m boundary NaN    : {stats['removed_boundary_nan']:,}")
    print(f"Final feature rows            : {stats['final_rows']:,}")
    print(f"Tickers in feature table      : {out['ticker'].nunique()}")
    print(f"Date range                    : {out['date'].min().date()} -> "
          f"{out['date'].max().date()}")
    print(f"\nRaw feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Rank columns          : {[c + '_rank' for c in FEATURE_COLS]}")
    print("\nAny NaN in output?    :", bool(out.isna().any().any()))
    print("\nhead():")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.head().to_string())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    build_features()
