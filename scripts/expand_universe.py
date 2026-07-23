#!/usr/bin/env python3
"""
Universe expansion — Phase 16. S&P 500 → US mid + large cap, WITH a revalidated model.

The point of this phase is that widening the universe is not a config change. Every feature
is a cross-sectional rank *within the universe*, so the frozen S&P 500 ranker's inputs mean
something different the moment the cross-section changes. Scoring the old model on the new
universe would produce confident, meaningless ranks. So this script retrains — same
architecture, same hyper-parameters, same leakage controls (expanding walk-forward, 21-day
embargo, never shuffled) — and produces a SECOND frozen model. The S&P 500 model is left
exactly where it is.

Stages (each cached on disk; re-run any stage with --stage):

    universe → panel → features → labels → model → report

    python scripts/expand_universe.py --stage panel     # ~1200 tickers x 7y from yfinance
    python scripts/expand_universe.py                   # run everything still missing

Outputs `figures/lab/universe_expansion.md` + `figures/lab/universe_expansion.png`.

⚠️ EDUCATIONAL SIMULATION. The universe is a CURRENT market-cap screen applied to all
history, so it is survivorship-biased in exactly the way `universe.py` documents — more so
than the S&P 500 panel, because the $2B floor deletes precisely the names that fell through
it. Every number here is DIRECTIONAL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from signals.baseline_momentum import backtest_scores, compute_metrics, SIGNAL, HORIZON  # noqa: E402
from signals.lgbm_ranker import walk_forward, INITIAL_TRAIN, STEP, EMBARGO, PARAMS  # noqa: E402
from utils.sp500_data import SP500DataBuilder  # noqa: E402
from utils.sp500_features import build_features  # noqa: E402
from utils.sp500_labels import build_labels  # noqa: E402
import universe as uni  # noqa: E402

UNIVERSE_CSV = Path("data/universe_midlarge.csv")
PANEL = Path("data/midlarge_panel.parquet")
FEATURES = Path("data/midlarge_features.parquet")
LABELED = Path("data/midlarge_labeled.parquet")
OOS = Path("data/midlarge_oos.parquet")

SP500_LABELED = Path("data/sp500_labeled.parquet")
OUT_MD = Path("figures/lab/universe_expansion.md")
OUT_PNG = Path("figures/lab/universe_expansion.png")


# ------------------------------------------------------------------ stages
def stage_universe(force=False) -> pd.DataFrame:
    if UNIVERSE_CSV.exists() and not force:
        return pd.read_csv(UNIVERSE_CSV)
    res = uni.build_universe(out=UNIVERSE_CSV)
    print(json.dumps(res["stats"], indent=2))
    return res["universe"]


def stage_panel(univ: pd.DataFrame, force=False, years=7, batch_size=100) -> pd.DataFrame:
    """Download the daily OHLCV panel for the new universe.

    `SP500DataBuilder.build_panel` is reused verbatim — same batching, same retries, same
    tidy schema and quality report — so the new panel is constructed identically to the one
    the S&P 500 results rest on. Only the ticker list differs.
    """
    if PANEL.exists() and not force:
        p = pd.read_parquet(PANEL)
        p["date"] = pd.to_datetime(p["date"])
        return p
    builder = SP500DataBuilder(years=years, batch_size=batch_size)
    panel, report = builder.build_panel(univ[["ticker"]])
    panel.to_parquet(PANEL, index=False)
    print(f"panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"tickers dropped by download: {len(report.get('dropped', {}))}")
    return panel


def stage_features(force=False):
    if FEATURES.exists() and not force:
        return
    build_features(panel_path=str(PANEL), out_path=str(FEATURES))


def stage_labels(force=False):
    if LABELED.exists() and not force:
        return
    build_labels(features_path=str(FEATURES), panel_path=str(PANEL), out_path=str(LABELED))


def stage_model(force=False) -> pd.DataFrame:
    """Retrain the ranker on the new universe — same config, same embargo, new frozen model."""
    if OOS.exists() and not force:
        oos = pd.read_parquet(OOS)
        oos["date"] = pd.to_datetime(oos["date"])
        return oos
    df = pd.read_parquet(LABELED)
    df["date"] = pd.to_datetime(df["date"])
    oos = walk_forward(df)
    oos.to_parquet(OOS, index=False)
    return oos


# ------------------------------------------------------------------ evaluation
def evaluate(labeled_path: Path, oos: pd.DataFrame | None = None, label: str = "",
             oos_cache: Path | None = None) -> dict:
    """Model vs no-ML momentum baseline on one universe, identical machinery for both."""
    if oos is None and oos_cache is not None and oos_cache.exists():
        oos = pd.read_parquet(oos_cache)
        oos["date"] = pd.to_datetime(oos["date"])
    if oos is None:
        df = pd.read_parquet(labeled_path)
        df["date"] = pd.to_datetime(df["date"])
        oos = walk_forward(df)
        if oos_cache is not None:
            oos.to_parquet(oos_cache, index=False)
    dates = np.sort(oos["date"].unique())
    rebal = dates[::HORIZON]

    model_bt = backtest_scores(oos, "model_score", rebal)
    base_bt = backtest_scores(oos, SIGNAL, rebal)
    m = compute_metrics(model_bt["res"], model_bt["ic"])
    b = compute_metrics(base_bt["res"], base_bt["ic"])
    return {
        "label": label,
        "n_names": int(oos["ticker"].nunique()),
        "n_rows": int(len(oos)),
        "start": str(pd.Timestamp(dates[0]).date()),
        "end": str(pd.Timestamp(dates[-1]).date()),
        "model": m, "baseline": b,
        "model_res": model_bt["res"], "base_res": base_bt["res"],
        "model_spread": model_bt["decile_spread"], "base_spread": base_bt["decile_spread"],
    }


def make_chart(new: dict, old: dict, out: Path = OUT_PNG) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=False)
    for ax, r in zip(axes, [old, new]):
        # `backtest_scores` returns a DATE-INDEXED frame, so the curves plot straight off it.
        eq_m = (1 + r["model_res"]["net_ret"]).cumprod()
        eq_b = (1 + r["base_res"]["net_ret"]).cumprod()
        eq_x = (1 + r["model_res"]["bench_ret"]).cumprod()
        ax.plot(eq_m.index, eq_m.values, lw=2, color="#d62728",
                label=f"LGBMRanker — Sharpe {r['model']['sharpe']:.2f}")
        ax.plot(eq_b.index, eq_b.values, lw=1.8, ls="--", color="#1f77b4",
                label=f"12-1 momentum — Sharpe {r['baseline']['sharpe']:.2f}")
        ax.plot(eq_x.index, eq_x.values, lw=1.2, ls=":", color="#888888",
                label="Equal-weight universe")
        ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
        ax.set_title(f"{r['label']} — {r['n_names']} names")
        ax.set_ylabel("Growth of $1 (long/short deciles, after cost)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("RankAlpha Phase 16 — did the edge survive the wider universe?")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _row(m: dict) -> str:
    return (f"| {m['sharpe']:+.3f} | {m['mean_ic']:+.4f} | {m['ic_tstat']:+.2f} | "
            f"{m['max_drawdown']:.1%} | {m['total_return']:+.1%} | {m['avg_turnover']:.2f} |")


def write_report(new: dict, old: dict, stats: dict, out: Path = OUT_MD) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    d_sharpe = new["model"]["sharpe"] - new["baseline"]["sharpe"]
    d_sharpe_old = old["model"]["sharpe"] - old["baseline"]["sharpe"]

    L = ["# RankAlpha — universe expansion to mid + large cap (Phase 16)\n",
         "*Reproducible via `python scripts/expand_universe.py`. Educational SIMULATION — "
         "survivorship-biased universe, results DIRECTIONAL, no promised returns.*\n",
         "## 1. The universe\n",
         "| filter | names remaining |", "|---|---|"]
    for k, v in stats.items():
        if k.startswith("after ") or k in ("registrants_listed_with_shares",
                                           "with_market_data", "universe_size"):
            L.append(f"| {k} | {v:,} |")
    L += ["",
          f"- Median market cap **${stats['median_market_cap']/1e9:.1f}B**, floor "
          f"**${stats['min_market_cap']/1e9:.2f}B**, aggregate "
          f"**${stats['total_market_cap']/1e12:.1f}T**.",
          f"- Built {stats['built_as_of']} from SEC registrant + share-count data and a "
          f"yfinance price/liquidity screen. A liquidity cap kept the top "
          f"{stats.get('universe_size')} by dollar volume "
          f"({stats.get('liquidity_cap_dropped', 0)} names dropped by that cap alone).",
          f"- Panel actually downloaded: **{stats['panel_tickers']} tickers**, "
          f"{stats['panel_rows']:,} rows, {stats['panel_start']} → {stats['panel_end']}.",
          ""]

    L += ["## 2. Retrained model vs no-ML baseline\n",
          "Same architecture and hyper-parameters as the S&P 500 ranker "
          f"(`lambdarank`, {PARAMS['n_estimators']} trees, lr {PARAMS['learning_rate']}), "
          f"same walk-forward ({INITIAL_TRAIN}d initial train, {STEP}d step, {EMBARGO}d "
          "embargo, never shuffled). This is a NEW frozen model; the S&P 500 one is "
          "untouched.\n",
          "| universe | model | Sharpe | Rank IC | IC t | max DD | total ret | turnover |",
          "|---|---|---|---|---|---|---|---|"]
    for r in (old, new):
        L.append(f"| {r['label']} ({r['n_names']} names) | LGBMRanker " + _row(r["model"]))
        L.append(f"| {r['label']} ({r['n_names']} names) | 12-1 momentum " + _row(r["baseline"]))
    L += ["",
          f"- **Model − baseline Sharpe**: S&P 500 {d_sharpe_old:+.3f} → "
          f"mid+large {d_sharpe:+.3f}.",
          f"- OOS windows: S&P 500 {old['start']} → {old['end']}; "
          f"mid+large {new['start']} → {new['end']}.",
          ""]

    L += ["## 3. Did the edge survive?\n", _survival_verdict(new, old), ""]

    L += ["## 4. Benchmark\n",
          "The scorecard benchmark is the **equal-weight new universe** (mean realized "
          "forward return across eligible names), which is the right field for a mid+large "
          "cap book — an S&P 500 benchmark would be measuring against a different universe "
          "than the one traded. The long/short decile book is roughly market-neutral by "
          "construction, so the benchmark is a reference, not the thing being beaten.\n",
          "## 5. Pointing the pie engine at the new universe\n",
          "`portfolio.beta_engine.build_portfolio` and `portfolio.engine.score_book` are "
          "already path-parameterised, so #15's pie runs on this universe with no code "
          "change:\n",
          "```python",
          "from portfolio.beta_engine import build_portfolio",
          "build_portfolio(10_000, target_beta=1.0,",
          "                panel_path='data/midlarge_panel.parquet',",
          "                tickers_path='data/universe_midlarge.csv',",
          "                features_path='data/midlarge_features.parquet',",
          "                labeled_path='data/midlarge_labeled.parquet')",
          "```",
          "Note the sector column: `data/universe_midlarge.csv` carries no GICS sector "
          "(SEC does not publish one), so the pie's per-sector caps degrade to a single "
          "'?' bucket until a sector mapping is attached. That is a real gap, not a "
          "rounding detail — the ≤5-per-sector and 30%-per-sector constraints are inert "
          "without it.\n",
          "## 6. Caveats\n",
          "- **Survivorship is worse here, not better — in both directions.** A current $2B "
          "floor applied to all history *deletes* the names that fell below it, and "
          "*includes* from day one the names that started small and compounded past it. The "
          "second half is the more dangerous one for a long/short decile book: the top "
          "decile gets to hold 2019's future ten-baggers. The S&P 500 panel has the same "
          "disease; this universe has a much larger dose, which is the first thing to "
          "suspect about any headline number in §2.",
          "- **Foreign issuers are included** — US-listed ADRs (e.g. AMBEV, Abivax) pass an "
          "exchange + market-cap screen. A US-domicile filter would change the field.",
          "- **Short history for younger names**: features need 253 trading days, so the "
          "early cross-section is much smaller than the late one.",
          "- Educational SIMULATION. Not investment advice."]
    out.write_text("\n".join(L))
    return out


def _survival_verdict(new: dict, old: dict) -> str:
    """State survival from the numbers, reading Rank IC and Sharpe as different things.

    **Rank IC measures ranking SKILL** — the average cross-sectional correlation between
    score and realized forward return, across the whole cross-section. **Sharpe measures the
    PAYOFF of the tails.** They usually move together. When they diverge — Sharpe up, Rank IC
    down — the extra return is coming from a handful of extreme names in the top decile
    rather than from ordering the field better, and on a universe screened by TODAY's market
    cap that is exactly the shape survivorship-inclusion bias takes: a name that was tiny in
    2019 and grew tenfold is in the panel from day one, and a momentum-and-volatility ranker
    finds it. So we report the divergence rather than banking the Sharpe.
    """
    d_new = new["model"]["sharpe"] - new["baseline"]["sharpe"]
    d_old = old["model"]["sharpe"] - old["baseline"]["sharpe"]
    ic_new, ic_t = new["model"]["mean_ic"], new["model"]["ic_tstat"]
    ic_old = old["model"]["mean_ic"]

    parts = [
        f"On the wider universe the ranker posts Sharpe {new['model']['sharpe']:+.2f} and "
        f"Rank IC {ic_new:+.4f} (t={ic_t:+.2f}) against a no-ML momentum baseline of "
        f"Sharpe {new['baseline']['sharpe']:+.2f} / IC {new['baseline']['mean_ic']:+.4f}.",
    ]

    ic_fell = ic_new < ic_old
    if d_new > 0.05 and ic_new > 0 and not ic_fell:
        parts.append(f"**The edge survived.** The model still beats its own baseline "
                     f"({d_new:+.3f} Sharpe vs {d_old:+.3f} on the S&P 500) and its ranking "
                     f"skill held up (Rank IC {ic_old:+.4f} → {ic_new:+.4f}) on a bigger, "
                     f"messier cross-section — which is the harder test.")
    elif d_new > 0.05 and ic_new > 0 and ic_fell:
        parts.append(
            f"**The edge survived on paper — but do NOT bank this number.** Model − baseline "
            f"Sharpe went {d_old:+.3f} → {d_new:+.3f}, yet Rank IC *fell* "
            f"{ic_old:+.4f} → {ic_new:+.4f} and got less significant "
            f"(t {old['model']['ic_tstat']:+.2f} → {ic_t:+.2f}). Those two facts point in "
            f"opposite directions: the model's ability to ORDER the cross-section got worse "
            f"while the payoff of its top decile got much better. That is what "
            f"survivorship-inclusion bias looks like on a universe screened by today's "
            f"market cap — names that were small in 2019 and compounded their way past the "
            f"$2B floor are present for the whole history, and a volatility-and-momentum "
            f"ranker concentrates in exactly them. Rank IC is the bias-resistant measure "
            f"here, and Rank IC says the skill did not improve. Treat the S&P 500 model as "
            f"the shipped one; treat this universe as a demonstration that the pipeline "
            f"retrains cleanly, not as evidence of a bigger edge.")
    elif ic_new > 0 and d_new > -0.05:
        parts.append(f"**The edge roughly survived, weakened.** Model − baseline Sharpe went "
                     f"{d_old:+.3f} → {d_new:+.3f}: the ranking information is still there "
                     f"(positive Rank IC) but it no longer buys a clear margin over plain "
                     f"momentum in the wider field.")
    else:
        parts.append(f"**The edge did NOT survive.** Model − baseline Sharpe went "
                     f"{d_old:+.3f} → {d_new:+.3f}. The ranker's advantage over plain "
                     f"momentum was specific to the S&P 500 cross-section it was designed "
                     f"on, and it does not transfer to mid caps. The honest conclusion is to "
                     f"keep the S&P 500 model as the shipped one.")
    parts.append("Rank IC t-statistics on ~4 years of monthly rebalances are not strong "
                 "evidence either way; treat a small Sharpe difference as noise, and a large "
                 "one as a question about the data before it is an answer about the model.")
    return " ".join(parts)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="RankAlpha universe expansion (#16)")
    ap.add_argument("--stage", default=None,
                    choices=["universe", "panel", "features", "labels", "model", "report"],
                    help="run only this stage (all earlier stages must already be cached)")
    ap.add_argument("--force", action="store_true", help="rebuild the chosen stage")
    ap.add_argument("--years", type=int, default=7)
    args = ap.parse_args()

    only = args.stage
    univ = stage_universe(force=args.force and only in (None, "universe"))
    if only == "universe":
        return 0

    stage_panel(univ, force=args.force and only == "panel", years=args.years)
    if only == "panel":
        return 0

    stage_features(force=args.force and only == "features")
    if only == "features":
        return 0

    stage_labels(force=args.force and only == "labels")
    if only == "labels":
        return 0

    oos = stage_model(force=args.force and only == "model")
    if only == "model":
        return 0

    print("\nEvaluating new universe…")
    new = evaluate(LABELED, oos, label="Mid + large cap")
    print("Evaluating S&P 500 with identical machinery (for comparison)…")
    old = evaluate(SP500_LABELED, None, label="S&P 500",
                   oos_cache=Path("data/sp500_oos_walkforward.parquet"))

    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    stats = {k: v for k, v in _universe_stats(univ).items()}
    stats.update({"panel_tickers": int(panel["ticker"].nunique()),
                  "panel_rows": int(len(panel)),
                  "panel_start": str(panel["date"].min().date()),
                  "panel_end": str(panel["date"].max().date())})

    make_chart(new, old)
    path = write_report(new, old, stats)
    print("\n" + _survival_verdict(new, old))
    print(f"\nReport: {path}  ·  Chart: {OUT_PNG}")
    return 0


def _universe_stats(univ: pd.DataFrame) -> dict:
    """Recover the filter funnel from the cached universe CSV if the build stats are gone."""
    cache = Path("data/universe_stats.json")
    if cache.exists():
        return json.loads(cache.read_text())
    return {"universe_size": len(univ),
            "median_market_cap": float(univ["market_cap"].median()),
            "min_market_cap": float(univ["market_cap"].min()),
            "total_market_cap": float(univ["market_cap"].sum()),
            "built_as_of": str(pd.Timestamp.today().date())}


if __name__ == "__main__":
    raise SystemExit(main())
