"""
RankAlpha — live paper-trading track record. Phase 9.

⚠️ EDUCATIONAL SIMULATION ONLY — NOT investment advice, NO real money, NO broker.
This is a FORWARD, OUT-OF-SAMPLE paper-trading log: we freeze the ranker at a single
training cutoff and watch it trade forward on data it has NEVER seen — exactly like real
paper trading (you ship a model once; you do NOT get to refit it with hindsight at every
step). Past paper-trading results do NOT predict future returns.

Design
------
* FREEZE the frozen-config LightGBM ranker ONCE on all labeled data up to `FREEZE_DATE`.
  No tuning, no new features — same `PARAMS`/`FEATURES` as the research model. The fitted
  model is cached to disk so a track update is cheap (one fit, then instant scoring).
* INCEPTION = first monthly rebalance at/after `FREEZE_DATE + EMBARGO` trading days, so no
  holding window (t → t+21) overlaps a training label (labels look 21d forward). Every
  recorded month is therefore strictly out-of-sample relative to the frozen model.
* At each rebalance `t` (every 21 trading days ≈ monthly) we build the SAME product book
  as the live engine — top-`TOP_N` long-only by model score, inverse-`vol_6m` weights,
  capped at `MAX_WEIGHT`, vol-targeted to `TARGET_VOL` (Balanced 14%) with a cash buffer —
  then MARK it to the realized next-month return `fwd_ret_1m`, net of 10 bps/side turnover.
  Cash earns 0 (an honest drag from the conservative vol target).
* Benchmark = equal-weight investable universe (mean `fwd_ret_1m` across eligible names),
  the same benchmark the Phase 5/6 evaluation uses (no SPY series in the panel).

This is SEPARATE from, and must never be blended with, the in-sample research backtest in
`signals/`. That backtest is the evidence base; this is the forward, honest track record.

Public API
----------
    update_track(as_of=None) -> {portfolio, holdings, stats, figures, n_new}
        Idempotent: computes any due rebalances (<= as_of) not already in the ledger and
        appends them. Re-running adds nothing if already up to date.
    load_track() -> (portfolio_df, holdings_df) | (None, None)
    compute_stats(portfolio_df) -> dict
"""

import logging
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from signals.lgbm_ranker import _fit_fold, FEATURES, EMBARGO  # noqa: E402
from signals.baseline_momentum import HORIZON, PERIODS_PER_YEAR  # noqa: E402
from portfolio.engine import _cap_weights, _book_vol, CACHE_DIR, DISCLAIMER  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("paper_trade")

FIG_DIR = Path("figures")
HOLDINGS_PATH = Path("data/paper_track_holdings.parquet")     # one row per (month, holding)
PORTFOLIO_PATH = Path("data/paper_track_portfolio.parquet")   # one row per month

# --- frozen-model + book config (matches the live engine's Balanced product book) ---
FREEZE_DATE = pd.Timestamp("2024-05-15")   # model frozen here; everything after is OOS
TOP_N = 50
MAX_WEIGHT = 0.08
TARGET_VOL = 0.14                          # Balanced (matches app default)
COST_PER_SIDE = 0.001                      # 10 bps/side on turnover


# --------------------------------------------------------------- frozen model
def _frozen_model(labeled: pd.DataFrame, freeze_date: pd.Timestamp):
    """Fit the frozen-config ranker ONCE on all data <= freeze_date. Cached to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"frozen_model_{pd.Timestamp(freeze_date).date()}.joblib"
    if cache_file.exists():
        logger.info("Loading frozen model from %s (no refit)", cache_file)
        return joblib.load(cache_file)

    train = labeled[labeled["date"] <= freeze_date].sort_values(["date", "ticker"])
    logger.info("Freezing model: train <= %s (%d rows, %d dates)",
                freeze_date.date(), len(train), train["date"].nunique())
    model = _fit_fold(train)
    joblib.dump(model, cache_file)
    logger.info("Cached frozen model to %s", cache_file)
    return model


def _rebalance_dates(labeled: pd.DataFrame, freeze_date: pd.Timestamp) -> np.ndarray:
    """Monthly (21-trading-day) rebalances strictly after the embargo boundary.

    Last train label looks to freeze_date+21d, so the first holding window must start at
    freeze_date + EMBARGO days. Step by HORIZON to tile non-overlapping monthly periods.
    """
    dates = np.sort(labeled["date"].unique())
    start_idx = int(np.searchsorted(dates, freeze_date))          # first date > freeze
    incept_idx = start_idx + EMBARGO
    return dates[incept_idx::HORIZON]


# ------------------------------------------------------------------- one month
def _build_and_mark(model, labeled, panel, t, prev_w: pd.Series):
    """Score the frozen model at t, build the product book, mark to realized fwd return.

    Returns (portfolio_row: dict, holding_rows: list[dict], weights: pd.Series).
    Point-in-time: scores/sizing use only info known at t; the mark uses fwd_ret_1m(t).
    """
    day = labeled[labeled["date"] == t].copy()
    day["model_score"] = model.predict(day[FEATURES])
    ranked = day.sort_values("model_score", ascending=False)

    holdings = ranked.head(TOP_N)                          # LONG-ONLY top names
    inv = 1.0 / holdings["vol_6m"]
    base = pd.Series((inv / inv.sum()).values, index=holdings["ticker"].values)
    capped = _cap_weights(base, MAX_WEIGHT)                # fully-invested (sum=1)

    book_vol = _book_vol(capped.index, capped, panel, t)
    k = min(1.0, TARGET_VOL / book_vol) if book_vol > 0 else 1.0
    weights = capped * k                                   # vol-targeted; rest is cash
    cash_w = float(1.0 - weights.sum())

    fwd = holdings.set_index("ticker")["fwd_ret_1m"]
    gross = float((weights * fwd.reindex(weights.index)).sum())   # cash earns 0

    # turnover vs the previous rebalance's invested weights (first build is from cash)
    tk = prev_w.index.union(weights.index)
    turnover = float((weights.reindex(tk, fill_value=0.0)
                      - prev_w.reindex(tk, fill_value=0.0)).abs().sum())
    cost = COST_PER_SIDE * turnover
    net = gross - cost

    bench = float(day["fwd_ret_1m"].mean())                # equal-weight universe

    p_row = {
        "date": pd.Timestamp(t),
        "net_ret": net,
        "gross_ret": gross,
        "cost": cost,
        "turnover": turnover,
        "bench_ret": bench,
        "invested_frac": float(weights.sum()),
        "cash_frac": cash_w,
        "book_vol": float(book_vol),
        "n_holdings": int(len(weights)),
    }
    h_rows = [{
        "date": pd.Timestamp(t),
        "ticker": tkr,
        "weight": float(w),
        "vol_6m": float(holdings.set_index("ticker").loc[tkr, "vol_6m"]),
        "model_score": float(holdings.set_index("ticker").loc[tkr, "model_score"]),
        "fwd_ret_1m": float(fwd.get(tkr, np.nan)),
    } for tkr, w in weights.sort_values(ascending=False).items()]

    return p_row, h_rows, weights


# --------------------------------------------------------------------- ledger
def load_track():
    """Return (portfolio_df, holdings_df) or (None, None) if no ledger yet."""
    if PORTFOLIO_PATH.exists() and HOLDINGS_PATH.exists():
        pf = pd.read_parquet(PORTFOLIO_PATH).sort_values("date").reset_index(drop=True)
        hd = pd.read_parquet(HOLDINGS_PATH).sort_values(["date", "weight"],
                                                        ascending=[True, False])
        return pf, hd
    return None, None


def update_track(as_of=None,
                 labeled_path="data/sp500_labeled.parquet",
                 panel_path="data/sp500_panel.parquet",
                 make_figures=True):
    """Append any due monthly rebalances (<= as_of) not already in the ledger. Idempotent.

    Re-running with the same data adds nothing. `as_of` defaults to the latest date with a
    realized fwd_ret_1m, so a fresh call records the whole forward track in one pass.
    """
    labeled = pd.read_parquet(labeled_path); labeled["date"] = pd.to_datetime(labeled["date"])
    panel = pd.read_parquet(panel_path); panel["date"] = pd.to_datetime(panel["date"])

    as_of = pd.Timestamp(as_of) if as_of is not None else labeled["date"].max()
    all_rebals = _rebalance_dates(labeled, FREEZE_DATE)
    due = [pd.Timestamp(d) for d in all_rebals if pd.Timestamp(d) <= as_of]

    pf_old, hd_old = load_track()
    done = set(pf_old["date"]) if pf_old is not None else set()
    missing = [d for d in due if d not in done]

    if not missing:
        logger.info("Track already up to date (%d months through %s); nothing to do.",
                    len(done), as_of.date())
        stats = compute_stats(pf_old) if pf_old is not None else {}
        figs = _figures(pf_old, stats) if (pf_old is not None and make_figures) else {}
        return {"portfolio": pf_old, "holdings": hd_old, "stats": stats,
                "figures": figs, "n_new": 0}

    model = _frozen_model(labeled, FREEZE_DATE)

    # previous invested weights for turnover continuity across incremental updates
    prev_w = pd.Series(dtype=float)
    if hd_old is not None and len(hd_old):
        last_done = hd_old[hd_old["date"] < missing[0]]
        if len(last_done):
            ld = last_done["date"].max()
            slc = last_done[last_done["date"] == ld]
            prev_w = pd.Series(slc["weight"].values, index=slc["ticker"].values)

    p_rows, h_rows = [], []
    for t in missing:
        p_row, h, prev_w = _build_and_mark(model, labeled, panel, t, prev_w)
        p_rows.append(p_row); h_rows.extend(h)
        logger.info("Marked %s: net %+.2f%% (bench %+.2f%%), invested %.0f%%, %d names",
                    t.date(), p_row["net_ret"] * 100, p_row["bench_ret"] * 100,
                    p_row["invested_frac"] * 100, p_row["n_holdings"])

    pf_new = pd.DataFrame(p_rows)
    hd_new = pd.DataFrame(h_rows)
    pf = (pd.concat([pf_old, pf_new]) if pf_old is not None else pf_new)
    hd = (pd.concat([hd_old, hd_new]) if hd_old is not None else hd_new)
    pf = pf.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    hd = (hd.drop_duplicates(["date", "ticker"])
            .sort_values(["date", "weight"], ascending=[True, False]).reset_index(drop=True))

    HOLDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pf.to_parquet(PORTFOLIO_PATH); hd.to_parquet(HOLDINGS_PATH)
    logger.info("Ledger now %d months (%s -> %s); added %d.",
                len(pf), pf["date"].min().date(), pf["date"].max().date(), len(missing))

    stats = compute_stats(pf)
    figs = _figures(pf, stats) if make_figures else {}
    return {"portfolio": pf, "holdings": hd, "stats": stats,
            "figures": figs, "n_new": len(missing)}


# ----------------------------------------------------------------- statistics
def compute_stats(pf: pd.DataFrame) -> dict:
    """Realized paper-trading performance — book and equal-weight benchmark."""
    r = pf["net_ret"].to_numpy()
    b = pf["bench_ret"].to_numpy()
    n = len(r)

    def _block(x):
        eq = float(np.prod(1 + x))
        ann_ret = eq ** (PERIODS_PER_YEAR / n) - 1 if n else float("nan")
        vol = float(np.std(x, ddof=1)) if n > 1 else float("nan")
        ann_vol = vol * np.sqrt(PERIODS_PER_YEAR)
        sharpe = (np.mean(x) / vol * np.sqrt(PERIODS_PER_YEAR)) if vol and vol > 0 else float("nan")
        equity = np.cumprod(1 + x)
        max_dd = float((equity / np.maximum.accumulate(equity) - 1).min()) if n else float("nan")
        return {"total_return": eq - 1, "ann_return": float(ann_ret),
                "ann_vol": float(ann_vol), "sharpe": float(sharpe),
                "max_drawdown": max_dd, "hit_rate": float(np.mean(x > 0))}

    book, bench = _block(r), _block(b)
    return {
        "inception": str(pf["date"].min().date()),
        "latest": str(pf["date"].max().date()),
        "n_months": int(n),
        "freeze_date": str(FREEZE_DATE.date()),
        "target_vol": TARGET_VOL,
        "book": book,
        "benchmark": bench,
        "excess_ann_return": book["ann_return"] - bench["ann_return"],
        "is_short_sample": n < 24,
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------- figures
def _figures(pf: pd.DataFrame, stats: dict) -> dict:
    FIG_DIR.mkdir(exist_ok=True)
    paths = {}
    dates = pd.to_datetime(pf["date"])
    book_eq = (1 + pf["net_ret"]).cumprod()
    bench_eq = (1 + pf["bench_ret"]).cumprod()
    dd = book_eq / book_eq.cummax() - 1
    b = stats["book"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 1], sharex=True)
    ax1.plot(dates, book_eq, lw=2, color="#2ca02c",
             label=f"RankAlpha paper book — Sharpe {b['sharpe']:.2f}, "
                   f"ann {b['ann_return']*100:+.1f}%")
    ax1.plot(dates, bench_eq, lw=1.6, ls="--", color="#888888",
             label=f"Equal-weight universe — ann {stats['benchmark']['ann_return']*100:+.1f}%")
    ax1.axhline(1.0, color="black", lw=0.6, alpha=0.5)
    ax1.axvline(FREEZE_DATE, color="#d62728", lw=1.0, ls=":",
                label=f"model frozen {FREEZE_DATE.date()}")
    short = "  ⚠ SHORT SAMPLE — not yet statistically meaningful" if stats["is_short_sample"] else ""
    ax1.set_title(f"RankAlpha — REALIZED PAPER-TRADING TRACK RECORD (out-of-sample)\n"
                  f"{stats['inception']} → {stats['latest']} · {stats['n_months']} months{short}\n"
                  f"{DISCLAIMER}", fontsize=9)
    ax1.set_ylabel("Growth of $1"); ax1.legend(loc="upper left", fontsize=8); ax1.grid(alpha=0.25)

    ax2.fill_between(dates, dd * 100, 0, color="#d62728", alpha=0.35)
    ax2.set_ylabel("Drawdown %"); ax2.set_xlabel("Date"); ax2.grid(alpha=0.25)
    paths["equity"] = str(FIG_DIR / "paper_track_equity.png")
    fig.tight_layout(); fig.savefig(paths["equity"], dpi=120); plt.close(fig)

    # honest stats card
    fig, ax = plt.subplots(figsize=(8, 5)); ax.axis("off")
    lines = [
        "RANKALPHA — PAPER-TRADING TRACK RECORD (realized, out-of-sample)",
        "",
        f"Model frozen on        : {stats['freeze_date']}  (never refit afterwards)",
        f"Track window           : {stats['inception']} -> {stats['latest']}  "
        f"({stats['n_months']} monthly rebalances)",
        f"Vol target (Balanced)  : {stats['target_vol']*100:.0f}%",
        "",
        f"{'':<22}{'PAPER BOOK':>12}{'EQ-WT BENCH':>14}",
        f"{'Total return':<22}{b['total_return']*100:>11.1f}%{stats['benchmark']['total_return']*100:>13.1f}%",
        f"{'Annualized return':<22}{b['ann_return']*100:>11.1f}%{stats['benchmark']['ann_return']*100:>13.1f}%",
        f"{'Annualized vol':<22}{b['ann_vol']*100:>11.1f}%{stats['benchmark']['ann_vol']*100:>13.1f}%",
        f"{'Sharpe (rf=0)':<22}{b['sharpe']:>12.2f}{stats['benchmark']['sharpe']:>14.2f}",
        f"{'Max drawdown':<22}{b['max_drawdown']*100:>11.1f}%{stats['benchmark']['max_drawdown']*100:>13.1f}%",
        f"{'Hit rate (months +)':<22}{b['hit_rate']*100:>11.0f}%{stats['benchmark']['hit_rate']*100:>13.0f}%",
        "",
    ]
    if stats["is_short_sample"]:
        lines.append(f"⚠ ONLY {stats['n_months']} MONTHS — too short to be statistically")
        lines.append("  meaningful. Do NOT read a Sharpe into this yet.")
    lines += ["", "REALIZED forward results — NOT the in-sample backtest, NOT a forecast."]
    lines += textwrap.wrap(DISCLAIMER, width=66)
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    paths["stats"] = str(FIG_DIR / "paper_track_stats.png")
    fig.tight_layout(); fig.savefig(paths["stats"], dpi=120); plt.close(fig)

    logger.info("Saved figures: %s", ", ".join(paths.values()))
    return paths


# --------------------------------------------------------------------- CLI
def _report(stats: dict):
    b = stats["book"]
    print("\n" + "=" * 72)
    print("RANKALPHA — PAPER-TRADING TRACK RECORD (REALIZED, OUT-OF-SAMPLE)")
    print("=" * 72)
    print(f"Model frozen {stats['freeze_date']} | track {stats['inception']} -> "
          f"{stats['latest']} | {stats['n_months']} months | vol target "
          f"{stats['target_vol']*100:.0f}%")
    print("-" * 72)
    print(f"{'':<20}{'PAPER BOOK':>12}{'EQ-WT BENCH':>14}")
    print(f"{'Total return':<20}{b['total_return']*100:>11.1f}%{stats['benchmark']['total_return']*100:>13.1f}%")
    print(f"{'Annualized return':<20}{b['ann_return']*100:>11.1f}%{stats['benchmark']['ann_return']*100:>13.1f}%")
    print(f"{'Annualized vol':<20}{b['ann_vol']*100:>11.1f}%{stats['benchmark']['ann_vol']*100:>13.1f}%")
    print(f"{'Sharpe (rf=0)':<20}{b['sharpe']:>12.2f}{stats['benchmark']['sharpe']:>14.2f}")
    print(f"{'Max drawdown':<20}{b['max_drawdown']*100:>11.1f}%{stats['benchmark']['max_drawdown']*100:>13.1f}%")
    print(f"{'Hit rate (months+)':<20}{b['hit_rate']*100:>11.0f}%{stats['benchmark']['hit_rate']*100:>13.0f}%")
    print("-" * 72)
    if stats["is_short_sample"]:
        print(f"⚠ ONLY {stats['n_months']} MONTHS of out-of-sample paper trading — too short "
              "to be statistically meaningful. No Sharpe-flexing on a small sample.")
    print(DISCLAIMER)
    print("=" * 72 + "\n")


if __name__ == "__main__":
    out = update_track()
    _report(out["stats"])
    print("figures:", ", ".join(out["figures"].values()))
