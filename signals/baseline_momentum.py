"""
No-ML baseline — RankAlpha Phase 4.

Pure 12-1 cross-sectional momentum, ZERO learning. This is the number the Phase 5
ranking model must beat (after-cost Sharpe + Rank IC).

Strategy
--------
* Universe / data: `data/sp500_labeled.parquet` (eligible stock-days with features +
  fwd_ret_1m, the realized 21-day-forward return).
* Rebalance every 21 trading days (≈ monthly), non-overlapping holding periods that
  tile the timeline (matches the 21-day label window — no overlap, no double-count).
* At each rebalance date t: rank eligible stocks by `mom_12_1m`; LONG the top decile
  (signal rank > 0.9), SHORT the bottom decile (signal rank <= 0.1).
* Sizing: within each leg, weight inverse to `vol_6m` (calmer stock → bigger weight),
  normalized so the long book = 100% and the short book = 100% (dollar-neutral,
  constant 100%/100% gross). Weights are stored SIGNED (long +, short −).
* Period return = Σ_i w_signed_i · fwd_ret_1m_i  (a short profits when fwd_ret < 0).

Costs / turnover
----------------
10 bps PER SIDE on turnover. Turnover at rebalance t is measured as
    turnover_t = Σ_i | w_{i,t} − w_{i,t-1} |   (signed target weights),
which sums every one-way buy and sell, so "10 bps per side" maps exactly onto it:
    cost_t = 0.001 × turnover_t.
Assumption: weights RESET to target at each rebalance (no intra-period drift modeled);
this is slightly conservative on turnover. The first rebalance builds from cash
(turnover = gross = 2.0 → 20 bps).

No train/test split / no embargo here
-------------------------------------
The baseline fits NOTHING: the signal (`mom_12_1m`) is known at t and the return is
measured t→t+21, so every period is already out-of-sample. The 21-day embargo only
matters in Phase 5, when a model is trained on past dates and tested on future ones.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("baseline_momentum")

HORIZON = 21                     # trading days per holding period
PERIODS_PER_YEAR = 252 / HORIZON  # ≈ 12 monthly periods
DECILE = 0.10                    # top/bottom 10%
COST_PER_SIDE = 0.001            # 10 bps
SIGNAL = "mom_12_1m"
SIZER = "vol_6m"


def _leg_weights(sub: pd.DataFrame, sign: float) -> pd.Series:
    """Inverse-vol weights for one leg, normalized to |sum| = 1, signed."""
    inv_vol = 1.0 / sub[SIZER]
    w = inv_vol / inv_vol.sum()
    return pd.Series(sign * w.values, index=sub["ticker"].values)


def rebalance_dates(df: pd.DataFrame) -> np.ndarray:
    """Every HORIZON-th unique trading day → non-overlapping monthly rebalances."""
    dates = np.sort(df["date"].unique())
    return dates[::HORIZON]


def backtest_scores(df: pd.DataFrame, score_col: str, rebal_dates) -> dict:
    """Run the long/short decile portfolio on an arbitrary score column.

    IDENTICAL machinery for both the no-ML baseline (score_col='mom_12_1m') and the
    Phase 5 model (score_col='model_score'). Higher score = more attractive (long).
    Returns dict with res (per-period frame), ic (Series), decile_spread (Series).
    """
    prev_w = pd.Series(dtype=float)   # signed target weights from the last rebalance
    records, ic_list, decile_rows = [], [], []

    for t in rebal_dates:
        day = df[df["date"] == t]
        if len(day) < 20:             # need a meaningful cross-section
            continue

        # Rank IC over the FULL cross-section (Spearman = Pearson on ranks).
        ic_list.append(day[score_col].corr(day["fwd_ret_1m"], method="spearman"))

        # Score-decile spread (sort by the SCORE, look at realized future return).
        d = day.assign(
            sig_decile=pd.qcut(day[score_col].rank(method="first"), 10, labels=False)
        )
        decile_rows.append(d[["sig_decile", "fwd_ret_1m"]])

        # Long top decile / short bottom decile by score percentile rank.
        srank = day[score_col].rank(pct=True)
        longs = day[srank > 1 - DECILE]
        shorts = day[srank <= DECILE]
        if longs.empty or shorts.empty:
            continue

        w = pd.concat([_leg_weights(longs, +1.0), _leg_weights(shorts, -1.0)])

        # Gross period return = signed-weighted realized forward return.
        fwd = day.set_index("ticker")["fwd_ret_1m"]
        gross_ret = float((w * fwd.reindex(w.index)).sum())

        # Turnover vs previous target weights (union of tickers).
        all_tk = prev_w.index.union(w.index)
        turnover = float((w.reindex(all_tk, fill_value=0.0)
                          - prev_w.reindex(all_tk, fill_value=0.0)).abs().sum())
        cost = COST_PER_SIDE * turnover
        prev_w = w

        records.append({
            "date": pd.Timestamp(t), "gross_ret": gross_ret, "cost": cost,
            "net_ret": gross_ret - cost, "turnover": turnover,
            "bench_ret": float(day["fwd_ret_1m"].mean()),  # equal-weight market proxy
            "n_long": len(longs), "n_short": len(shorts),
        })

    res = pd.DataFrame(records).set_index("date")
    ic = pd.Series(ic_list).dropna()
    decile_spread = pd.concat(decile_rows).groupby("sig_decile")["fwd_ret_1m"].mean()
    return {"res": res, "ic": ic, "decile_spread": decile_spread}


def run_backtest(labeled_path="data/sp500_labeled.parquet"):
    df = pd.read_parquet(labeled_path)
    df["date"] = pd.to_datetime(df["date"])

    rebal = rebalance_dates(df)
    logger.info("%d trading days -> %d rebalances every %d days",
                df["date"].nunique(), len(rebal), HORIZON)

    bt = backtest_scores(df, SIGNAL, rebal)
    res, ic, decile_spread = bt["res"], bt["ic"], bt["decile_spread"]

    metrics = compute_metrics(res, ic)
    _plot_equity(res, metrics, out="data/baseline_equity_curve.png")
    _print_report(res, metrics, decile_spread, ic)
    return res, metrics, decile_spread


def compute_metrics(res: pd.DataFrame, ic: pd.Series) -> dict:
    r = res["net_ret"]
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min())

    bench_equity = (1 + res["bench_ret"]).cumprod()

    n = len(ic)
    ic_t = ic.mean() / ic.std(ddof=1) * np.sqrt(n) if n > 1 else np.nan

    total_ret = float(equity.iloc[-1] - 1)
    ann_ret = float(equity.iloc[-1] ** (PERIODS_PER_YEAR / len(r)) - 1)

    return {
        "n_rebalances": len(r),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "mean_ic": float(ic.mean()),
        "ic_tstat": float(ic_t),
        "ic_n": n,
        "avg_turnover": float(res["turnover"].mean()),
        "avg_cost_bps": float(res["cost"].mean() * 1e4),
        "bench_total_return": float(bench_equity.iloc[-1] - 1),
        "equity": equity,
        "bench_equity": bench_equity,
    }


def _plot_equity(res, metrics, out="data/baseline_equity_curve.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(metrics["equity"].index, metrics["equity"].values,
            label=f"12-1 momentum L/S (after cost) — Sharpe {metrics['sharpe']:.2f}",
            lw=2, color="#1f77b4")
    ax.plot(metrics["bench_equity"].index, metrics["bench_equity"].values,
            label="Equal-weight universe (market proxy)",
            lw=1.5, color="#888888", ls="--")
    ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
    ax.set_title("RankAlpha Phase 4 — No-ML 12-1 Momentum Baseline (after 10bps/side costs)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved equity curve -> %s", out)


def _print_report(res, metrics, decile_spread, ic):
    print("\n" + "=" * 64)
    print("PHASE 4 — NO-ML 12-1 MOMENTUM BASELINE (after 10 bps/side costs)")
    print("=" * 64)
    print(f"Date range            : {res.index.min().date()} -> {res.index.max().date()}")
    print(f"Rebalances            : {metrics['n_rebalances']} (every {HORIZON} trading days)")
    print(f"Avg long / short names: {res['n_long'].mean():.0f} / {res['n_short'].mean():.0f}")
    print("-" * 64)
    print(f"Annualized Sharpe     : {metrics['sharpe']:.3f}      <-- number ML must beat")
    print(f"Max drawdown          : {metrics['max_drawdown']*100:.2f}%")
    print(f"Total return (L/S)    : {metrics['total_return']*100:+.2f}%")
    print(f"Annualized return     : {metrics['ann_return']*100:+.2f}%")
    print(f"Mean Rank IC          : {metrics['mean_ic']:+.4f}   <-- number ML must beat")
    print(f"Rank IC t-stat        : {metrics['ic_tstat']:+.2f} (n={metrics['ic_n']})")
    print(f"Avg turnover / rebal  : {metrics['avg_turnover']:.2f}  "
          f"(avg cost {metrics['avg_cost_bps']:.1f} bps/rebal)")
    print(f"Benchmark total ret   : {metrics['bench_total_return']*100:+.2f}% "
          f"(equal-weight universe)")
    print("-" * 64)
    print("Signal-decile spread — sort by mom_12_1m, mean realized fwd_ret_1m:")
    print("  (the REAL test: monotonic rise => momentum has predictive content)")
    for d, v in decile_spread.items():
        bar = "#" * int(max(0, v * 200))
        print(f"   decile {int(d)} : {v*100:+.3f}%  {bar}")
    spread = decile_spread.iloc[-1] - decile_spread.iloc[0]
    print(f"  D9 − D0 spread      : {spread*100:+.3f}% per 21d")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    run_backtest()
