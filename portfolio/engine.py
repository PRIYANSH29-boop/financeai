"""
RankAlpha pie engine — Phase 7. Risk-managed, self-explaining, LONG-ONLY portfolio.

⚠️ EDUCATIONAL SIMULATION ONLY — NOT investment advice, NOT a live product, NO real
money, and NO fabricated return probabilities. Past backtest ≠ future returns.

Public API
----------
    build_portfolio(amount, date=None, top_n=50, target_vol=0.14, max_weight=0.08)
        -> {as_of, weights, dollar_allocations, explanations, risk_stats, figures}

This is what the Phase 9 product page calls. The L/S research backtest
(`signals/lgbm_ranker.py`, `signals/evaluate.py`) is the evidence base and stays intact;
this engine is the long-only *product* on top of the SAME frozen model.

The model is UNTOUCHED: we import the frozen config (PARAMS, FEATURES) and fit it at a
production cutoff (all labeled data up to `as_of − 21 trading days`, the same embargo),
then score the latest cross-section on/before `as_of`. No tuning, no new features.
"""

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from signals.lgbm_ranker import _fit_fold, FEATURES, EMBARGO  # noqa: E402
from signals.baseline_momentum import backtest_scores, PERIODS_PER_YEAR, HORIZON  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pie_engine")

FIG_DIR = Path("figures")
CACHE_DIR = Path("data/cache")          # gitignored; persists the fitted book across runs
BUNDLE_DIR = Path("portfolio/bundle")   # TRACKED; the committed hosted-demo bundle
BUNDLE_BOOK = BUNDLE_DIR / "score_book.joblib"   # frozen fitted book (no parquets needed)
DISCLAIMER = ("EDUCATIONAL SIMULATION — NOT investment advice. No real money. "
              "Backtested on survivorship-biased data; past backtest does NOT predict "
              "future returns.")

# Human-readable factor descriptions for the self-explanation layer.
FACTOR_DESC = {
    "vol_6m_rank": "6-month volatility",
    "size_rank": "size proxy (log price)",
    "mom_12_1m_rank": "12-1 month momentum",
    "mom_6m_rank": "6-month momentum",
    "mom_3m_rank": "3-month momentum",
    "reversal_1m_rank": "1-month reversal",
    "liquidity_rank": "liquidity (volume surge)",
}


# --------------------------------------------------------------------- scoring
def _score_universe(as_of, feats, labeled):
    """Fit the frozen model up to (as_of − EMBARGO) and score the as_of cross-section."""
    feat_dates = np.sort(feats["date"].unique())
    if as_of not in feat_dates:
        as_of = feat_dates[feat_dates <= as_of][-1]   # latest available on/before
    cutoff = feat_dates[max(0, np.searchsorted(feat_dates, as_of) - EMBARGO)]

    train = labeled[labeled["date"] <= cutoff].sort_values(["date", "ticker"])
    logger.info("Production fit: train <= %s (%d rows) | embargo %dd | score %s",
                pd.Timestamp(cutoff).date(), len(train), EMBARGO, pd.Timestamp(as_of).date())
    model = _fit_fold(train)

    cross = feats[feats["date"] == as_of].copy()
    cross["model_score"] = model.predict(cross[FEATURES])
    return model, cross.sort_values("model_score", ascending=False), as_of


# --------------------------------------------------------------- sizing helpers
def _cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Cap each weight at `cap`, redistribute excess to uncapped names. Sum stays 1."""
    w = w / w.sum()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def _book_vol(tickers, weights, panel, as_of, lookback=126) -> float:
    """Annualized portfolio vol from the trailing daily-return covariance of holdings."""
    hist = panel[(panel["ticker"].isin(tickers)) & (panel["date"] <= as_of)]
    wide = (hist.pivot_table(index="date", columns="ticker", values="adj_close")
                .sort_index().tail(lookback + 1))
    rets = wide.pct_change().dropna(how="all")
    rets = rets[list(weights.index)]                 # align column order to weights
    cov = rets.cov().to_numpy()
    w = weights.to_numpy()
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(252))


# ----------------------------------------------------------- risk evidence base
def _oos_long_only_stats(labeled, cache="data/oos_long_only.parquet"):
    """Backtest the LONG-ONLY top-decile model book on the walk-forward OOS (cached)."""
    cache = Path(cache)
    if cache.exists():
        res = pd.read_parquet(cache)
    else:
        from signals.lgbm_ranker import walk_forward, _rebal_dates
        oos = walk_forward(labeled)
        rebal = _rebal_dates(oos)
        # Long-only top decile: reuse machinery, then keep only the long leg's P&L by
        # scoring a long-only book here (top decile inverse-vol, no short).
        res = _long_only_backtest(oos, rebal)
        res.to_parquet(cache)
    r = res["net_ret"]
    equity = (1 + r).cumprod()
    dd = float((equity / equity.cummax() - 1).min())
    ann_vol = float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    return {
        "ann_vol": ann_vol,
        "max_drawdown": dd,
        "monthly_ret_p05": float(r.quantile(0.05)),
        "monthly_ret_p95": float(r.quantile(0.95)),
        "monthly_ret_worst": float(r.min()),
        "n_periods": len(r),
        "oos_start": str(res.index.min().date()),
        "oos_end": str(res.index.max().date()),
    }


def _long_only_backtest(oos, rebal, cap=0.08):
    """Top-decile long-only inverse-vol book; 10bps/side one-way cost on turnover."""
    prev_w = pd.Series(dtype=float)
    recs = []
    for t in rebal:
        day = oos[oos["date"] == t]
        if len(day) < 20:
            continue
        srank = day["model_score"].rank(pct=True)
        longs = day[srank > 0.9]
        if longs.empty:
            continue
        inv = 1.0 / longs["vol_6m"]
        w = pd.Series((inv / inv.sum()).values, index=longs["ticker"].values)
        w = _cap_weights(w, cap)
        fwd = day.set_index("ticker")["fwd_ret_1m"]
        gross = float((w * fwd.reindex(w.index)).sum())
        tk = prev_w.index.union(w.index)
        turn = float((w.reindex(tk, fill_value=0) - prev_w.reindex(tk, fill_value=0)).abs().sum())
        prev_w = w
        recs.append({"date": pd.Timestamp(t), "net_ret": gross - 0.001 * turn})
    return pd.DataFrame(recs).set_index("date")


# ----------------------------------------------------------------- explanations
def _explain(model, holdings):
    """Per-holding SHAP top contributors + raw rank features."""
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(holdings[FEATURES])
    out = {}
    for i, (_, row) in enumerate(holdings.iterrows()):
        contrib = pd.Series(sv[i], index=FEATURES).sort_values(key=np.abs, ascending=False)
        top = contrib.head(3)
        reasons = []
        for f, c in top.items():
            direction = "↑" if c > 0 else "↓"
            reasons.append(f"{FACTOR_DESC[f]} {direction} (rank {row[f]:.2f})")
        out[row["ticker"]] = {
            "reasons": reasons,
            "ranks": {f: round(float(row[f]), 3) for f in FEATURES},
        }
    return out


# ------------------------------------------------------------------- figures
def _figures(holdings, weights, cash_w, explanations, risk_stats, amount):
    FIG_DIR.mkdir(exist_ok=True)
    paths = {}

    # (a) allocation pie
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = list(weights.index) + (["CASH"] if cash_w > 1e-6 else [])
    sizes = list(weights.values) + ([cash_w] if cash_w > 1e-6 else [])
    ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
           textprops={"fontsize": 8})
    ax.set_title(f"RankAlpha pie — ${amount:,.0f} ({len(weights)} holdings + cash)\n"
                 f"{DISCLAIMER}", fontsize=9)
    paths["pie"] = str(FIG_DIR / "portfolio_pie.png")
    fig.tight_layout(); fig.savefig(paths["pie"], dpi=120); plt.close(fig)

    # (b) per-holding factor exposures (top 12 holdings, 7 rank features)
    topn = weights.head(12).index
    mat = holdings.set_index("ticker").loc[topn, FEATURES]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([f.replace("_rank", "") for f in FEATURES], rotation=45, ha="right")
    ax.set_yticks(range(len(topn))); ax.set_yticklabels(topn)
    ax.set_title("Per-holding factor exposures (rank 0–1; green = high)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="cross-sectional rank")
    paths["factors"] = str(FIG_DIR / "portfolio_factors.png")
    fig.tight_layout(); fig.savefig(paths["factors"], dpi=120); plt.close(fig)

    # (c) risk summary card
    fig, ax = plt.subplots(figsize=(8, 5)); ax.axis("off")
    lines = [
        "RANKALPHA — HONEST RISK PANEL (backtested, long-only book)",
        "",
        f"Backtested annualized volatility : {risk_stats['ann_vol']*100:.1f}%",
        f"Historical max drawdown          : {risk_stats['max_drawdown']*100:.1f}%",
        f"Monthly return range (5–95 pct)  : {risk_stats['monthly_ret_p05']*100:+.1f}%"
        f" to {risk_stats['monthly_ret_p95']*100:+.1f}%",
        f"Worst monthly return (observed)  : {risk_stats['monthly_ret_worst']*100:+.1f}%",
        f"Evidence window                  : {risk_stats['oos_start']} → {risk_stats['oos_end']}"
        f" ({risk_stats['n_periods']} months)",
        "",
        "This is a HISTORICAL RANGE, not a forecast or a probability.",
        DISCLAIMER,
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=10)
    paths["risk"] = str(FIG_DIR / "portfolio_risk.png")
    fig.tight_layout(); fig.savefig(paths["risk"], dpi=120); plt.close(fig)

    logger.info("Saved figures: %s", ", ".join(paths.values()))
    return paths


# --------------------------------------------------------------------- API
def _cache_key(date, top_n, max_weight, *data_paths) -> str:
    """Stable key for the score_book output: params + input-file mtimes (so the cache
    self-invalidates when the underlying data changes)."""
    parts = [str(date), str(top_n), str(max_weight)]
    for p in data_paths:
        pth = Path(p)
        parts.append(f"{p}:{pth.stat().st_mtime_ns if pth.exists() else 0}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def score_book(date=None, top_n=50, max_weight=0.08,
               features_path="data/sp500_features.parquet",
               labeled_path="data/sp500_labeled.parquet",
               panel_path="data/sp500_panel.parquet",
               tickers_path="data/sp500_tickers.csv",
               use_cache=True):
    """Expensive, target-vol-INDEPENDENT half: fit the frozen model, pick the long book,
    cap weights, estimate book vol, build explanations + base risk stats.

    Cache this (it does not depend on amount or target_vol) so the risk slider — which
    only changes vol-target scaling — stays instant. `vol_target_scale_k`/cash come later.

    The fitted result is also persisted to disk via joblib (`data/cache/`), so a fresh
    `streamlit run` LOADS the book in milliseconds instead of refitting the model.

    HOSTED PATH: on a free cloud host the source parquets are absent (gitignored, and we
    never download 500 tickers). When `features_path` does not exist we serve the committed
    precomputed bundle (`portfolio/bundle/score_book.joblib`) — a fully self-contained
    frozen book that `finalize_portfolio` consumes directly, so the app boots with no refit.
    Regenerate the bundle with `python -m portfolio.make_bundle`.
    """
    if not Path(features_path).exists() and BUNDLE_BOOK.exists():
        logger.info("Source parquets absent; loading committed bundle %s (no refit)",
                    BUNDLE_BOOK)
        return joblib.load(BUNDLE_BOOK)

    if use_cache:
        key = _cache_key(date, top_n, max_weight, features_path, labeled_path,
                         panel_path, tickers_path)
        cache_file = CACHE_DIR / f"score_book_{key}.joblib"
        if cache_file.exists():
            logger.info("Loading cached book from %s (no refit)", cache_file)
            return joblib.load(cache_file)

    feats = pd.read_parquet(features_path); feats["date"] = pd.to_datetime(feats["date"])
    labeled = pd.read_parquet(labeled_path); labeled["date"] = pd.to_datetime(labeled["date"])
    panel = pd.read_parquet(panel_path); panel["date"] = pd.to_datetime(panel["date"])
    meta = pd.read_csv(tickers_path).set_index("ticker")

    as_of = pd.Timestamp(date) if date is not None else feats["date"].max()
    model, ranked, as_of = _score_universe(as_of, feats, labeled)

    holdings = ranked.head(top_n).copy()             # LONG-ONLY top names
    logger.info("Selected top %d of %d names (long-only) as of %s",
                len(holdings), len(ranked), pd.Timestamp(as_of).date())

    inv = 1.0 / holdings["vol_6m"]
    base = pd.Series((inv / inv.sum()).values, index=holdings["ticker"].values)
    capped = _cap_weights(base, max_weight)
    book_vol = _book_vol(capped.index, capped, panel, as_of)

    explanations = _explain(model, holdings)
    for tk in explanations:
        explanations[tk]["sector"] = str(meta.loc[tk, "sector"]) if tk in meta.index else "?"

    risk_base = _oos_long_only_stats(labeled)

    out = {
        "as_of": str(pd.Timestamp(as_of).date()),
        "holdings": holdings,
        "capped": capped,                 # fully-invested fractional weights (sum=1)
        "book_vol": float(book_vol),
        "explanations": explanations,
        "risk_base": risk_base,
    }

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(out, cache_file)
        logger.info("Cached fitted book to %s", cache_file)

    return out


def cash_fraction(book_vol: float, target_vol: float) -> float:
    """Instant: cash fraction the vol-target implies (never levers). For the live slider."""
    k = min(1.0, target_vol / book_vol) if book_vol > 0 else 1.0
    return float(1.0 - k)


def finalize_portfolio(book: dict, amount: float, target_vol: float = 0.14,
                       make_figures: bool = True) -> dict:
    """Cheap, target-vol-DEPENDENT half: apply vol-target scaling -> dollars -> figures."""
    capped, book_vol = book["capped"], book["book_vol"]
    k = min(1.0, target_vol / book_vol) if book_vol > 0 else 1.0
    weights = (capped * k).sort_values(ascending=False)
    cash_w = float(1.0 - weights.sum())

    dollar_alloc = (weights * amount).round(2).to_dict()
    dollar_alloc["CASH"] = round(cash_w * amount, 2)

    risk = dict(book["risk_base"])
    risk["target_vol"] = target_vol
    risk["estimated_book_vol_preTarget"] = round(book_vol, 4)
    risk["vol_target_scale_k"] = round(k, 3)
    risk["invested_fraction"] = round(float(weights.sum()), 4)
    risk["disclaimer"] = DISCLAIMER

    figures = (_figures(book["holdings"], weights, cash_w, book["explanations"], risk, amount)
               if make_figures else {})

    return {
        "as_of": book["as_of"],
        "amount": amount,
        "weights": weights.round(4).to_dict(),
        "cash_weight": round(cash_w, 4),
        "dollar_allocations": dollar_alloc,
        "explanations": book["explanations"],
        "risk_stats": risk,
        "figures": figures,
    }


def build_portfolio(amount, date=None, top_n=50, target_vol=0.14, max_weight=0.08,
                    **paths):
    """Full pipeline (score_book + finalize_portfolio). Same return shape as before."""
    book = score_book(date=date, top_n=top_n, max_weight=max_weight, **paths)
    return finalize_portfolio(book, amount, target_vol)


def _demo():
    p = build_portfolio(10_000)
    print("\n" + "=" * 70)
    print(f"RANKALPHA PIE — ${p['amount']:,} as of {p['as_of']}  (LONG-ONLY, simulated)")
    print("=" * 70)
    r = p["risk_stats"]
    print(f"Vol target {r['target_vol']*100:.0f}% | est. book vol {r['estimated_book_vol_preTarget']*100:.1f}%"
          f" -> scale k={r['vol_target_scale_k']} | cash {p['cash_weight']*100:.1f}%")
    print(f"Risk (backtested long-only, {r['oos_start']}→{r['oos_end']}): "
          f"ann vol {r['ann_vol']*100:.1f}%, max DD {r['max_drawdown']*100:.1f}%, "
          f"monthly 5–95% {r['monthly_ret_p05']*100:+.1f}..{r['monthly_ret_p95']*100:+.1f}%")
    print("-" * 70)
    print(f"{'Ticker':<8}{'$ alloc':>10}{'wt':>7}   why (top SHAP factors)")
    for tk, w in list(p["weights"].items())[:12]:
        ex = p["explanations"][tk]
        print(f"{tk:<8}{p['dollar_allocations'][tk]:>10,.0f}{w*100:>6.1f}%   "
              f"{ex['reasons'][0]}; {ex['reasons'][1]}")
    print(f"{'CASH':<8}{p['dollar_allocations']['CASH']:>10,.0f}{p['cash_weight']*100:>6.1f}%")
    print("-" * 70)
    print("figures:", ", ".join(p["figures"].values()))
    print(DISCLAIMER)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    _demo()
