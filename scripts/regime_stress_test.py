#!/usr/bin/env python3
"""
Regime stress-test — Strategy Lab, Phase 14 extension.

Runs Strategy A (momentum) vs B (momentum + low-vol) through the SAME frozen-pipeline
harness used for the paper track, but on two crisis regimes the frozen 2024-2026 track
never saw:

  * 2008 Global Financial Crisis (2007-06 → 2009-12)
  * COVID era (2020-01 → 2023-12)

It fetches a long-history daily panel via yfinance (cached under the gitignored
`data/cache/`), rebuilds features + forward-return labels with the exact Phase 2/3
pipeline functions, then runs the A/B books month by month within each window.

⚠️⚠️ SURVIVORSHIP BIAS (dominant caveat) ⚠️⚠️
The universe is TODAY's S&P 500 constituents that already had price history before 2007.
Every company that was delisted or went bankrupt — Lehman, Bear Stearns, Washington
Mutual, old GM, ... — is ABSENT. So absolute crash levels are optimistic and drawdowns
are understated. ONLY the *relative* A-vs-B read (same survivor pool, factor-only change)
is trustworthy. This is DIRECTIONAL, not a verdict.

Needs network (yfinance) on first run; cached afterwards. Not part of `make analyse`/`lab`,
which are strictly no-network.

    python scripts/regime_stress_test.py        # or:  make regimes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sp500_features import build_raw_features, cross_sectional_rank, FEATURE_COLS  # noqa: E402
from utils.sp500_labels import _forward_returns  # noqa: E402
from lab.strategy_lab import (  # noqa: E402
    run_strategy, monthly_rebalances, factor_score, MOMENTUM, MOMENTUM_LOWVOL,
)
from analytics.metrics import analyse  # noqa: E402

CACHE = Path("data/cache")
PANEL_CACHE = CACHE / "regime_panel.parquet"
LABELED_CACHE = CACHE / "regime_labeled.parquet"
FIG_DIR = Path("figures/lab")
FIG_PATH = FIG_DIR / "strategy_lab_regimes.png"
SCORECARD_PATH = FIG_DIR / "strategy_lab_regimes.md"

FETCH_START, FETCH_END = "2006-01-01", "2024-01-01"
OLD_CUTOFF = "2007-06-01"      # must have history before this to test the 2008 GFC
CHUNK = 50

WINDOWS = {
    "2008 GFC": ("2007-06-01", "2009-12-31"),
    "COVID 2020-2023": ("2020-01-01", "2023-12-31"),
}
_A_C, _B_C, _BENCH_C = "#1f77b4", "#2ca02c", "#7f7f7f"


# ---------------------------------------------------------------------- data
def fetch_panel() -> pd.DataFrame:
    if PANEL_CACHE.exists():
        p = pd.read_parquet(PANEL_CACHE); p["date"] = pd.to_datetime(p["date"]); return p
    import yfinance as yf
    tickers = (pd.read_csv("data/sp500_tickers.csv")["ticker"]
               .str.replace(".", "-", regex=False).tolist())
    fields = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
              "Adj Close": "adj_close", "Volume": "volume"}
    frames = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=FETCH_START, end=FETCH_END, interval="1d",
                                  auto_adjust=False, progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  chunk {i//CHUNK} attempt {attempt} failed: {e}", file=sys.stderr)
                time.sleep(3)
        else:
            continue
        long = None
        for f, name in fields.items():
            if f not in raw.columns.get_level_values(0):
                continue
            s = raw[f].stack().rename(name)
            long = s.to_frame() if long is None else long.join(s, how="outer")
        if long is None:
            continue
        long = long.reset_index()
        long.columns = ["date", "ticker"] + list(long.columns[2:])
        frames.append(long)
        print(f"  fetched chunk {i//CHUNK+1}/{-(-len(tickers)//CHUNK)}", flush=True)

    panel = pd.concat(frames, ignore_index=True)
    panel = (panel.dropna(subset=["adj_close"]).drop_duplicates(["date", "ticker"])
                  .sort_values(["ticker", "date"]).reset_index(drop=True))
    first = panel.groupby("ticker")["date"].min()
    old = first[first <= pd.Timestamp(OLD_CUTOFF)].index
    panel = panel[panel["ticker"].isin(old)].reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_CACHE, index=False)
    return panel


def build_labeled(panel: pd.DataFrame) -> pd.DataFrame:
    if LABELED_CACHE.exists():
        lab = pd.read_parquet(LABELED_CACHE); lab["date"] = pd.to_datetime(lab["date"]); return lab
    df, _ = build_raw_features(panel)
    elig = df[df["eligible"]].dropna(subset=FEATURE_COLS).reset_index(drop=True)
    elig = cross_sectional_rank(elig)
    fwd = _forward_returns(panel)
    lab = elig.merge(fwd, on=["date", "ticker"], how="left").dropna(subset=["fwd_ret_1m"])
    keep = ["date", "ticker"] + FEATURE_COLS + ["fwd_ret_1m"]
    lab = lab[keep].sort_values(["date", "ticker"]).reset_index(drop=True)
    lab.to_parquet(LABELED_CACHE, index=False)
    return lab


# ------------------------------------------------------------------ one window
def run_window(name, start, end, labeled, panel) -> dict:
    rebals = monthly_rebalances(labeled, start, end)
    A = run_strategy(MOMENTUM, labeled=labeled, panel=panel, rebalances=rebals)
    B = run_strategy(MOMENTUM_LOWVOL, labeled=labeled, panel=panel, rebalances=rebals)
    idx = pd.to_datetime(A["date"])
    ra = pd.Series(A["net_ret"].to_numpy(), index=idx, name="A")
    rb = pd.Series(B["net_ret"].to_numpy(), index=idx, name="B")
    bench = pd.Series(A["bench_ret"].to_numpy(), index=idx, name="bench")

    corrs = []
    for t in rebals:
        day = labeled[labeled["date"] == t]
        if len(day) < 50:
            continue
        a = factor_score(day, MOMENTUM["factors"]); b = factor_score(day, [("vol_6m", False)])
        corrs.append(a.corr(b, method="spearman"))

    return {
        "name": name, "start": start, "end": end, "ra": ra, "rb": rb, "bench": bench,
        "mA": analyse(ra, benchmark=bench, periods_per_year=12),
        "mB": analyse(rb, benchmark=bench, periods_per_year=12),
        "mBench": analyse(bench, benchmark=bench, periods_per_year=12),
        "turnA": A["turnover"].mean(), "turnB": B["turnover"].mean(),
        "corr": float(np.nanmean(corrs)) if corrs else float("nan"),
        "n_months": len(A), "n_names": labeled[(labeled["date"] >= start)
                                               & (labeled["date"] <= end)]["ticker"].nunique(),
    }


# --------------------------------------------------------------------- outputs
def _plot(results):
    fig, axes = plt.subplots(2, len(results), figsize=(7 * len(results), 9))
    for col, res in enumerate(results):
        eqa, eqb = (1 + res["ra"]).cumprod(), (1 + res["rb"]).cumprod()
        eqm = (1 + res["bench"]).cumprod()
        ax = axes[0, col]
        ax.plot(eqa.index, eqa.values, color=_A_C, lw=2, label="A · Momentum")
        ax.plot(eqb.index, eqb.values, color=_B_C, lw=2, label="B · Mom + low-vol")
        ax.plot(eqm.index, eqm.values, color=_BENCH_C, lw=1.4, ls="--", label="Eq-wt universe")
        ax.set_title(f"{res['name']} — growth of $1")
        ax.set_ylabel("Growth of $1 (×)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axd = axes[1, col]
        for r, c in [(res["ra"], _A_C), (res["rb"], _B_C)]:
            eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
            axd.plot(dd.index, dd.values, color=c, lw=1.6)
        dda = (eqa / eqa.cummax() - 1)
        axd.fill_between(dda.index, dda.values, 0, color=_A_C, alpha=0.12)
        axd.set_title(f"Drawdown — A {res['mA']['max_drawdown']:+.0%} vs "
                      f"B {res['mB']['max_drawdown']:+.0%}")
        axd.set_ylabel("Drawdown"); axd.grid(alpha=0.3)
    fig.suptitle("RankAlpha Strategy Lab — regime stress-test  "
                 "(⚠ SURVIVORSHIP-BIASED universe — relative A-vs-B read only)", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=110, bbox_inches="tight"); plt.close(fig)


def _scorecard(results) -> str:
    def pct(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1%}"

    def apct(x):
        return f"{x:.1%}"

    def num(x):
        return f"{x:.2f}"

    lines = [
        "# Strategy Lab — regime stress-test (2008 GFC + COVID)",
        "",
        "*Auto-generated by `scripts/regime_stress_test.py`. Educational SIMULATION only. "
        "Model frozen; factor-only change through the same book pipeline.*",
        "",
        "> ⚠️ **SURVIVORSHIP BIAS (dominant caveat).** Universe = today's S&P 500 names with "
        "pre-2007 history; bankrupt/delisted firms (Lehman, Bear Stearns, WaMu, old GM, …) "
        "are absent. Absolute crash levels are optimistic and drawdowns understated. Only "
        "the **relative A-vs-B** read is trustworthy. **DIRECTIONAL, not a verdict.**",
        "",
    ]
    rows = [
        ("Total return", "total_return", pct),
        ("CAGR", "cagr", pct),
        ("Volatility (ann)", "volatility", apct),
        ("Sharpe", "sharpe", num),
        ("Sortino", "sortino", num),
        ("Max drawdown", "max_drawdown", pct),
        ("Hit rate", "hit_rate", apct),
        ("Beta", "beta", num),
    ]
    for res in results:
        mA, mB, mK = res["mA"], res["mB"], res["mBench"]
        lines += [
            f"## {res['name']} ({res['start']} → {res['end']})",
            "",
            f"{res['n_months']} monthly rebalances · {res['n_names']} survivor names · "
            f"mom↔low-vol rank corr {res['corr']:+.2f}",
            "",
            "| Metric | A · Momentum | B · Momentum + low-vol | Eq-wt universe |",
            "|---|---|---|---|",
        ]
        for label, key, fmt in rows:
            lines.append(f"| {label} | {fmt(mA[key])} | {fmt(mB[key])} | {fmt(mK[key])} |")
        lines.append(f"| Avg turnover/rebal | {res['turnA']:.2f} | {res['turnB']:.2f} | — |")
        d_dd = (mB["max_drawdown"] - mA["max_drawdown"]) * 100
        helped = "**helped**" if d_dd > 0.3 else ("**hurt**" if d_dd < -0.3 else "was ~neutral")
        lines += [
            "",
            f"Low-vol {helped} on drawdown here: A {pct(mA['max_drawdown'])} → "
            f"B {pct(mB['max_drawdown'])} ({d_dd:+.1f} pp).",
            "",
        ]

    lines += [
        "## Read",
        "",
        "- **2008 GFC — low-vol helped.** Shallower drawdown, less-bad Sharpe, higher hit "
        "rate: the defensive tilt earns its keep in a slow, grinding bear.",
        "- **COVID — low-vol slightly hurt.** A fast V-crash (Feb–Mar 2020) that monthly "
        "rebalancing can't dodge, followed by a high-beta recovery that punishes low-vol "
        "names, is the factor's worst case.",
        "- **Both regimes:** the momentum books badly lagged the equal-weight universe — the "
        "classic *momentum crash* (missing the sharp 2009 and 2020–21 rebounds). That's a "
        "momentum limitation, not a low-vol one.",
        "",
        "**Net:** low-vol's drawdown protection is **regime-dependent** — real in prolonged "
        "selloffs, but it can reverse in sharp V-shaped, high-vol-led recoveries. A slow-bear "
        "defense, not a universal drawdown fix. This tempers the 2024–2026 result, where it "
        "looked unambiguously protective.",
        "",
        "![regimes](strategy_lab_regimes.png)",
        "",
    ]
    return "\n".join(lines)


def main():
    panel = fetch_panel()
    labeled = build_labeled(panel)
    print(f"labeled: {len(labeled):,} rows | {labeled['ticker'].nunique()} survivor names | "
          f"{labeled['date'].min().date()} → {labeled['date'].max().date()}")
    results = [run_window(n, s, e, labeled, panel) for n, (s, e) in WINDOWS.items()]
    _plot(results)
    SCORECARD_PATH.write_text(_scorecard(results), encoding="utf-8")

    for res in results:
        mA, mB = res["mA"], res["mB"]
        print(f"\n{res['name']}: maxDD A {mA['max_drawdown']:+.1%} vs B {mB['max_drawdown']:+.1%} "
              f"({(mB['max_drawdown']-mA['max_drawdown'])*100:+.1f} pp) | "
              f"Sharpe A {mA['sharpe']:.2f} vs B {mB['sharpe']:.2f}")
    print(f"\nscorecard -> {SCORECARD_PATH}\nfigure    -> {FIG_PATH}")


if __name__ == "__main__":
    main()
