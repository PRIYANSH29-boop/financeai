"""
Honest evaluation of the FROZEN Phase 5 ranker — RankAlpha Phase 6.

This module does NOT change the model, tune hyperparameters, or add features. It only
stress-tests the frozen walk-forward ranker and writes the limitations up. Touching the
model here would mean overfitting the test window.

Produces (all on the Phase 5 OOS window, after costs):
  1. Subperiod stability   — yearly Sharpe + mean Rank IC, model vs baseline.
  2. Rank IC time series    — per-rebalance IC, mean/std/t-stat/% positive, PNG.
  3. Decile monotonicity    — mean fwd_ret_1m by model-score decile, PNG.
  4. Cost sensitivity       — model vs baseline at 5/10/20/30 bps/side.
  5. Turnover / capacity     — model vs baseline.
  6. Feature importance      — avg LightGBM gain across folds + SHAP on a sample.
  7. Crash caveat            — OOS contains no major momentum crash (stated).
"""

import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from signals.baseline_momentum import (  # noqa: E402
    backtest_scores, compute_metrics, HORIZON, PERIODS_PER_YEAR, SIGNAL,
)
from signals.lgbm_ranker import walk_forward, FEATURES, _rebal_dates  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("evaluate")

COST_GRID = [0.0005, 0.0010, 0.0020, 0.0030]   # 5, 10, 20, 30 bps per side


def ic_timeseries(oos, rebal, score_col) -> pd.Series:
    """Per-rebalance Spearman IC between score and realized fwd_ret_1m, date-indexed."""
    out = {}
    for t in rebal:
        day = oos[oos["date"] == t]
        if len(day) >= 20:
            out[pd.Timestamp(t)] = day[score_col].corr(day["fwd_ret_1m"], method="spearman")
    return pd.Series(out).dropna()


def sharpe(net_ret: pd.Series) -> float:
    if net_ret.std(ddof=1) == 0 or len(net_ret) < 2:
        return float("nan")
    return float(net_ret.mean() / net_ret.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


# ----------------------------------------------------------------- 1. subperiods
def subperiod_stability(model_res, base_res, model_ic, base_ic):
    rows = []
    years = sorted(set(model_res.index.year))
    for y in years:
        mr = model_res[model_res.index.year == y]["net_ret"]
        br = base_res[base_res.index.year == y]["net_ret"]
        mic = model_ic[model_ic.index.year == y]
        bic = base_ic[base_ic.index.year == y]
        rows.append({
            "year": y, "n": len(mr),
            "model_sharpe": sharpe(mr), "base_sharpe": sharpe(br),
            "model_ic": mic.mean(), "base_ic": bic.mean(),
        })
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------- 4. cost sweep
def cost_sensitivity(oos, rebal):
    rows = []
    for c in COST_GRID:
        m = compute_metrics(backtest_scores(oos, "model_score", rebal, cost=c)["res"],
                            pd.Series([0.0]))
        b = compute_metrics(backtest_scores(oos, SIGNAL, rebal, cost=c)["res"],
                            pd.Series([0.0]))
        rows.append({
            "bps_per_side": int(c * 1e4),
            "model_sharpe": m["sharpe"], "base_sharpe": b["sharpe"],
            "edge": m["sharpe"] - b["sharpe"],
        })
    return pd.DataFrame(rows).set_index("bps_per_side")


# ----------------------------------------------------------- 6. feature import
def feature_importance(models):
    """Average LightGBM gain importance across folds, normalized to %."""
    gains = np.zeros(len(FEATURES))
    for m in models:
        gains += m.booster_.feature_importance(importance_type="gain")
    gains /= len(models)
    imp = pd.Series(gains, index=FEATURES).sort_values(ascending=False)
    return (imp / imp.sum() * 100).round(2)


def shap_importance(models, oos, n=3000):
    """Mean |SHAP| per feature on a sample, using the last fold's model."""
    try:
        import shap
    except Exception as e:  # noqa: BLE001
        logger.warning("SHAP unavailable (%s) — skipping", e)
        return None
    sample = oos[FEATURES].sample(min(n, len(oos)), random_state=42)
    expl = shap.TreeExplainer(models[-1])
    sv = expl.shap_values(sample)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    return (mean_abs / mean_abs.sum() * 100).round(2)


# ------------------------------------------------------------------- plots
def plot_ic_timeseries(model_ic, base_ic, out="data/eval_ic_timeseries.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(model_ic.index, model_ic.values, width=18, color="#d62728", alpha=0.7,
           label=f"Model IC (mean {model_ic.mean():+.3f})")
    ax.axhline(model_ic.mean(), color="#d62728", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Phase 6 — Per-rebalance Rank IC (model score vs fwd_ret_1m)")
    ax.set_ylabel("Spearman IC"); ax.set_xlabel("Rebalance date")
    ax.legend(loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    logger.info("Saved %s", out)


def plot_decile(model_spread, out="data/eval_decile_monotonicity.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if d == model_spread.index.min()
              else "#2ca02c" if d == model_spread.index.max() else "#1f77b4"
              for d in model_spread.index]
    ax.bar(model_spread.index, model_spread.values * 100, color=colors)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Phase 6 — Mean fwd_ret_1m by MODEL-score decile (OOS)")
    ax.set_ylabel("Mean fwd_ret_1m (%) per 21d"); ax.set_xlabel("Model-score decile (0=short, 9=long)")
    ax.set_xticks(range(10)); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    logger.info("Saved %s", out)


# ------------------------------------------------------------------- driver
def run(labeled_path="data/sp500_labeled.parquet"):
    df = pd.read_parquet(labeled_path)
    df["date"] = pd.to_datetime(df["date"])

    oos, models = walk_forward(df, return_models=True)
    rebal = _rebal_dates(oos)

    model_bt = backtest_scores(oos, "model_score", rebal)
    base_bt = backtest_scores(oos, SIGNAL, rebal)
    model_ic = ic_timeseries(oos, rebal, "model_score")
    base_ic = ic_timeseries(oos, rebal, SIGNAL)

    sub = subperiod_stability(model_bt["res"], base_bt["res"], model_ic, base_ic)
    cost = cost_sensitivity(oos, rebal)
    imp = feature_importance(models)
    shap_imp = shap_importance(models, oos)

    plot_ic_timeseries(model_ic, base_ic)
    plot_decile(model_bt["decile_spread"])

    _report(oos, rebal, model_bt, base_bt, model_ic, base_ic, sub, cost, imp, shap_imp)
    return {"sub": sub, "cost": cost, "imp": imp, "shap": shap_imp,
            "model_ic": model_ic, "base_ic": base_ic}


def _report(oos, rebal, model_bt, base_bt, model_ic, base_ic, sub, cost, imp, shap_imp):
    n = len(model_ic)
    pos = (model_ic > 0).mean() * 100
    t = model_ic.mean() / model_ic.std(ddof=1) * np.sqrt(n)
    m_turn = model_bt["res"]["turnover"].mean()
    b_turn = base_bt["res"]["turnover"].mean()

    print("\n" + "=" * 72)
    print("PHASE 6 — HONEST EVALUATION (frozen model, OOS after costs)")
    print("=" * 72)
    print(f"OOS {oos['date'].min().date()} -> {oos['date'].max().date()} | "
          f"{len(rebal)} rebalances")

    print("\n[1] SUBPERIOD STABILITY (yearly)")
    print(sub.round(3).to_string())

    print("\n[2] RANK IC TIME SERIES (model)")
    print(f"   mean {model_ic.mean():+.4f} | std {model_ic.std(ddof=1):.4f} | "
          f"t-stat {t:+.2f} | % positive {pos:.0f}% | n {n}")

    print("\n[4] COST SENSITIVITY (annualized Sharpe by bps/side)")
    print(cost.round(3).to_string())
    vanish = cost[cost["edge"] <= 0]
    if len(vanish):
        print(f"   -> model edge vanishes at {vanish.index[0]} bps/side")
    else:
        print(f"   -> model keeps its edge across all tested costs (up to "
              f"{cost.index.max()} bps/side)")

    print("\n[5] TURNOVER / CAPACITY")
    print(f"   model avg turnover {m_turn:.2f}/rebal | baseline {b_turn:.2f}/rebal "
          f"| ratio {m_turn / b_turn:.2f}x")

    print("\n[6] FEATURE IMPORTANCE — avg LightGBM gain across folds (%)")
    print(imp.to_string())
    if shap_imp is not None:
        print("\n    SHAP mean|value| on a sample (%):")
        print(shap_imp.to_string())
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run()
