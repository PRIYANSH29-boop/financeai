"""
Cross-sectional label builder — RankAlpha Phase 3 (LABELS).

Joins forward-return labels onto the Phase 2 feature table and writes
`data/sp500_labeled.parquet` for the ranking model.

This is Phase 3 only — NO model here (that's Phase 4).

Labels (holding period = monthly rebalance => horizon = 21 trading days):
  * fwd_ret_1m   — adj_close[t+21] / adj_close[t] - 1, per ticker (groupby shift(-21)).
  * label_rank   — within-date percentile rank of fwd_ret_1m, in (0, 1].
  * label_decile — within-date decile 0..9 (higher = better forward return), the graded
                   relevance label LightGBM lambdarank needs, with group = date.

==============================================================================
⚠️  EMBARGO RULE FOR PHASE 4+ (read before building any train/test split)
==============================================================================
The label is the ONLY column that looks into the future — by design. Because
`fwd_ret_1m` peeks 21 trading days ahead, any chronological split MUST embargo
**21 trading days** between the end of train and the start of test. Otherwise the
last ~21 training labels overlap the test window and leak look-ahead information
(this is the single-stock purge lesson, generalized to the panel). Phase 4 must
enforce a 21-day embargo (and purge) on every fold.
==============================================================================
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sp500_labels")

HORIZON = 21          # trading days (≈ 1 month) — monthly rebalance holding period.
N_DECILES = 10

LABEL_COLS = ["fwd_ret_1m", "label_rank", "label_decile"]


def _forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute fwd_ret_1m on the contiguous daily panel (per ticker, never across).

    Done on the panel rather than the feature table so the -21 shift runs over an
    unbroken daily series of adj_close.
    """
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    fwd = panel.groupby("ticker", sort=False)["adj_close"].shift(-HORIZON)
    panel["fwd_ret_1m"] = fwd / panel["adj_close"] - 1
    return panel[["date", "ticker", "fwd_ret_1m"]]


def build_labels(features_path="data/sp500_features.parquet",
                 panel_path="data/sp500_panel.parquet",
                 out_path="data/sp500_labeled.parquet"):
    feats = pd.read_parquet(features_path)
    feats["date"] = pd.to_datetime(feats["date"])
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    logger.info("Features: %d rows | Panel: %d rows", len(feats), len(panel))

    # Forward returns from the panel, merged onto the eligible feature rows.
    fwd = _forward_returns(panel)
    df = feats.merge(fwd, on=["date", "ticker"], how="left")

    # Drop the trailing ~21 days that have no future price.
    before = len(df)
    df = df.dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)
    removed_future_nan = before - len(df)

    # Within-date labels.
    g = df.groupby("date")["fwd_ret_1m"]
    df["label_rank"] = g.rank(pct=True)
    # Tie-break with method="first" so qcut produces equal-sized bins; labels 0..9 with
    # higher decile = higher forward return.
    df["label_decile"] = (
        df.groupby("date")["fwd_ret_1m"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), N_DECILES, labels=False))
        .astype("int8")
    )

    out = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    out_path = Path(out_path)
    out.to_parquet(out_path, index=False)
    logger.info("Saved labeled table -> %s (%d rows, %d cols)",
                out_path, len(out), len(out.columns))

    stats = {
        "rows_before_drop": before,
        "removed_future_nan": removed_future_nan,
        "final_rows": len(out),
    }
    print_report(out, stats)
    return out, stats


def print_report(out: pd.DataFrame, stats: dict):
    print("\n" + "=" * 60)
    print("LABEL BUILD SUMMARY — Phase 3")
    print("=" * 60)
    print(f"Rows before future-NaN drop : {stats['rows_before_drop']:,}")
    print(f"Removed — no future (t+21)  : {stats['removed_future_nan']:,}")
    print(f"Final labeled rows          : {stats['final_rows']:,}")
    print(f"Tickers                     : {out['ticker'].nunique()}")
    print(f"Date range                  : {out['date'].min().date()} -> "
          f"{out['date'].max().date()}")
    print(f"\nLabel columns: {LABEL_COLS}")
    print("Any NaN in output? :", bool(out.isna().any().any()))

    # Decile balance check on a sample mid-sample date.
    sample_date = out["date"].drop_duplicates().sort_values().iloc[len(out['date'].unique()) // 2]
    sub = out[out["date"] == sample_date]
    counts = sub["label_decile"].value_counts().sort_index()
    print(f"\nDecile balance on {sample_date.date()} "
          f"({len(sub)} stocks): expect ~{len(sub) / N_DECILES:.1f} each")
    print("  decile : count")
    for d, n in counts.items():
        print(f"     {d}    :  {n}")

    print("\nhead():")
    show = ["date", "ticker", "mom_12_1m_rank", "fwd_ret_1m", "label_rank", "label_decile"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out[show].head().to_string())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    build_labels()
