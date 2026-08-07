"""
LightGBM learning-to-rank model — RankAlpha Phase 5.

Trains an `LGBMRanker` (objective `lambdarank`) to score the S&P 500 cross-section each
day, then trades the model score with the SAME portfolio machinery as the Phase 4
baseline (`signals.baseline_momentum.backtest_scores`) so the comparison is
apples-to-apples on an identical out-of-sample window.

Bar to beat (Phase 4, full sample): after-cost Sharpe 0.40, mean Rank IC +0.019.

Walk-forward validation with embargo
------------------------------------
* Expanding window. Initial train = 504 trading days (~2y); step = 126 days (~6 months);
  retrain from scratch each fold.
* Fold k tests the date block [test_start, test_start + STEP). Training uses every date
  with index < test_start, MINUS a 21-trading-day EMBARGO: we drop training dates with
  index >= test_start - 21. Because a training label at date d looks 21 days forward
  (d+21), this guarantees d+21 < test_start — no training label overlaps the test block.
* NEVER shuffled (rows kept in date order; group = date).

Leakage guarantees
------------------
* No shuffling anywhere; chronological folds only.
* Per-fold embargo: model only ever sees data strictly before (test_start - 21d).
* Features are already point-in-time (within-date cross-sectional ranks from Phase 2).
* No global fit over the full sample — every fold fits on its own past slice only.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lightgbm import LGBMRanker  # noqa: E402

from signals.baseline_momentum import (  # noqa: E402
    backtest_scores, compute_metrics, HORIZON, SIGNAL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lgbm_ranker")

FEATURES = [
    "mom_3m_rank", "mom_6m_rank", "mom_12_1m_rank", "reversal_1m_rank",
    "vol_6m_rank", "liquidity_rank", "size_rank",
]
LABEL = "label_decile"

INITIAL_TRAIN = 504     # ~2 years of trading days before the first test block
STEP = 126              # ~6 months per walk-forward fold
EMBARGO = 21            # trading days dropped from train tail (label looks 21d forward)

# Modest, FIXED config — shallow trees, strong regularization, low LR. No tuning yet.
PARAMS = dict(
    objective="lambdarank",
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=15,
    max_depth=4,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def _fit_fold(train: pd.DataFrame) -> LGBMRanker:
    """Fit one ranker. group = number of rows per date (rows already sorted by date)."""
    group = train.groupby("date", sort=False).size().to_numpy()
    model = LGBMRanker(**PARAMS)
    model.fit(train[FEATURES], train[LABEL], group=group)
    return model


def walk_forward(df: pd.DataFrame, return_models: bool = False, fit_fold=None):
    """Expanding-window walk-forward with 21d embargo. Returns OOS rows + model_score.

    If return_models=True, also returns the list of per-fold fitted models (used by the
    Phase 6 evaluation for feature importance — no retraining needed).

    `fit_fold` overrides the estimator (#31 Arm 2). It takes the fold's training frame and
    returns anything with `.predict(X)`. Everything else — fold boundaries, the expanding
    window, the 21-day embargo, the group structure, the feature list, the label — stays in
    this one function, so a rival library runs the IDENTICAL protocol by construction rather
    than by a second implementation that has to be trusted to match.
    """
    fit_fold = fit_fold or _fit_fold
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    n = len(dates)

    oos_parts, models = [], []
    fold = 0
    test_start = INITIAL_TRAIN
    while test_start < n:
        test_end = min(test_start + STEP, n)
        train_cutoff = test_start - EMBARGO            # exclusive index into `dates`

        train_dates = dates[:train_cutoff]
        test_dates = dates[test_start:test_end]
        train = df[df["date"].isin(train_dates)]
        test = df[df["date"].isin(test_dates)].copy()

        model = fit_fold(train)
        test["model_score"] = model.predict(test[FEATURES])
        oos_parts.append(test)
        models.append(model)

        fold += 1
        logger.info(
            "fold %d | train <= %s (%d days, %d rows) | EMBARGO %dd | test %s..%s (%d days)",
            fold, pd.Timestamp(train_dates[-1]).date(), len(train_dates), len(train),
            EMBARGO, pd.Timestamp(test_dates[0]).date(),
            pd.Timestamp(test_dates[-1]).date(), len(test_dates),
        )
        test_start = test_end

    oos = pd.concat(oos_parts, ignore_index=True)
    logger.info("Walk-forward done: %d folds, OOS %s -> %s (%d rows)",
                fold, oos["date"].min().date(), oos["date"].max().date(), len(oos))
    return (oos, models) if return_models else oos


def _rebal_dates(oos: pd.DataFrame) -> np.ndarray:
    dates = np.sort(oos["date"].unique())
    return dates[::HORIZON]


def run(labeled_path="data/sp500_labeled.parquet"):
    df = pd.read_parquet(labeled_path)
    df["date"] = pd.to_datetime(df["date"])

    oos = walk_forward(df)
    rebal = _rebal_dates(oos)

    # Identical machinery for both — model score vs raw momentum, SAME OOS rebalances.
    model_bt = backtest_scores(oos, "model_score", rebal)
    base_bt = backtest_scores(oos, SIGNAL, rebal)
    model_m = compute_metrics(model_bt["res"], model_bt["ic"])
    base_m = compute_metrics(base_bt["res"], base_bt["ic"])

    _plot(model_bt["res"], model_m, base_bt["res"], base_m,
          out="data/ranker_vs_baseline.png")
    _report(oos, rebal, model_m, base_m,
            model_bt["decile_spread"], base_bt["decile_spread"])
    return model_m, base_m


def _plot(m_res, m_metrics, b_res, b_metrics, out="data/ranker_vs_baseline.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(m_metrics["equity"].index, m_metrics["equity"].values,
            label=f"LGBMRanker (OOS, after cost) — Sharpe {m_metrics['sharpe']:.2f}",
            lw=2, color="#d62728")
    ax.plot(b_metrics["equity"].index, b_metrics["equity"].values,
            label=f"12-1 momentum baseline (same OOS) — Sharpe {b_metrics['sharpe']:.2f}",
            lw=1.8, color="#1f77b4", ls="--")
    ax.plot(m_metrics["bench_equity"].index, m_metrics["bench_equity"].values,
            label="Equal-weight universe", lw=1.2, color="#888888", ls=":")
    ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
    ax.set_title("RankAlpha Phase 5 — LightGBM Ranker vs No-ML Baseline (OOS, after costs)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved comparison equity curve -> %s", out)


def _fmt_row(name, m):
    return (f"{name:<14}  Sharpe {m['sharpe']:+.3f} | RankIC {m['mean_ic']:+.4f} "
            f"(t={m['ic_tstat']:+.2f}) | maxDD {m['max_drawdown']*100:+.1f}% | "
            f"ret {m['total_return']*100:+.1f}%")


def _report(oos, rebal, model_m, base_m, model_spread, base_spread):
    print("\n" + "=" * 70)
    print("PHASE 5 — LIGHTGBM RANKER vs NO-ML BASELINE (OOS, after 10bps/side costs)")
    print("=" * 70)
    print(f"OOS window   : {oos['date'].min().date()} -> {oos['date'].max().date()}")
    print(f"Rebalances   : {model_m['n_rebalances']} (every {HORIZON} trading days)")
    print(f"Cadence      : expanding train, initial {INITIAL_TRAIN}d (~2y), "
          f"step {STEP}d (~6mo), {EMBARGO}d embargo")
    print("-" * 70)
    print(_fmt_row("LGBMRanker", model_m))
    print(_fmt_row("Baseline", base_m))
    print("-" * 70)
    d_sharpe = model_m["sharpe"] - base_m["sharpe"]
    d_ic = model_m["mean_ic"] - base_m["mean_ic"]
    print(f"Δ Sharpe (model − baseline) : {d_sharpe:+.3f}")
    print(f"Δ Rank IC (model − baseline): {d_ic:+.4f}")
    print("-" * 70)
    print("Model signal-decile spread (sort by MODEL score, mean realized fwd_ret_1m):")
    for d, v in model_spread.items():
        bar = "#" * int(max(0, v * 200))
        print(f"   decile {int(d)} : {v*100:+.3f}%  {bar}")
    m_spread = model_spread.iloc[-1] - model_spread.iloc[0]
    b_spread = base_spread.iloc[-1] - base_spread.iloc[0]
    print(f"  model D9−D0 : {m_spread*100:+.3f}% / 21d  (baseline {b_spread*100:+.3f}%)")
    print(f"  model short-leg (D0) fwd_ret : {model_spread.iloc[0]*100:+.3f}%  "
          f"(baseline {base_spread.iloc[0]*100:+.3f}%) "
          f"-- lower = better short leg")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run()
