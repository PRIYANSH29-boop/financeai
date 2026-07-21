"""
RankAlpha pie engine — Phase 15. BETA-targeted, sector-diversified, self-explaining,
LONG-ONLY portfolio.

⚠️ EDUCATIONAL SIMULATION ONLY — NOT investment advice, NOT a live product, NO real
money, and NO promised or fabricated returns. Past backtest ≠ future returns.

Public API
----------
    build_portfolio(capital, target_beta, benchmark="SPY", top_n=20)
        -> {as_of, capital, target_beta, achieved_beta, benchmark, weights,
            cash_weight, dollar_allocations, slices, scorecard, notes, figures}

Why a NEW module (deliberate deviation from the literal #15 signature).
----------------------------------------------------------------------
`portfolio/engine.py` already ships a `build_portfolio(amount, …, target_vol=…)` that
targets *volatility* and is wired into the Phase-9 product page + hosted bundle. #15 asks
for a *beta*-targeted pie with a different signature. Rather than mutate the frozen,
shipped vol engine, this module adds the beta variant on TOP of the same frozen model:
it reuses `engine.score_book()` for scoring / holdings / SHAP reasons / sectors and never
refits anything. The model stays frozen; this is pure measurement + portfolio construction.

How target beta is hit — honestly.
----------------------------------
Portfolio beta is *linear* in weights: β_portfolio = Σ wᵢ·βᵢ, and cash has β = 0. So
  • target ≤ book beta → add a CASH SLEEVE, scale k = target / β_book (never levers).
  • target > book beta → TILT weights toward higher-beta names (bounded, bisected). If the
    max long-only tilt still can't reach it, we CAP at the achievable beta and RESET the
    expectation in `notes` — we do NOT fabricate a pie that pretends to hit it.
Because βᵢ and the scorecard are computed on the SAME monthly window, the scorecard's
realised beta equals Σ wᵢβᵢ by construction (≈ target ✓). This is an in-sample match of
today's weights over history — a historical characterisation, NOT a forecast.

Benchmark note.
---------------
SPY/IVV/VOO are not in the committed panel and this path is offline-by-design (the repo's
no-network guardrail). So `benchmark="SPY"` resolves to an EQUAL-WEIGHT S&P-500 universe
proxy built from the panel — clearly a proxy for the market, not the real SPY ETF. If the
benchmark ticker were present in the panel it would be used directly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from portfolio.engine import score_book, _cap_weights, DISCLAIMER  # noqa: E402
from analytics.metrics import beta as _beta, analyse  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("beta_engine")

FIG_DIR = Path("figures/portfolio")
PPY = 12  # monthly

# construction knobs (documented defaults; not tuned)
NAME_CAP = 0.08          # ≤ 8% per single name — no landmine
SECTOR_CAP = 0.30        # ≤ 30% weight per sector — diversified
SECTOR_MAX_NAMES = 5     # ≤ 5 names per sector picked from the candidate pool
MIN_HISTORY = 36         # require ≥ 36 monthly obs so βᵢ is stable & the window is common
MAX_LOOKBACK = 72        # cap the scoring window at 6y of monthly returns


# --------------------------------------------------------------- returns helpers
def _monthly_returns(panel: pd.DataFrame, tickers=None) -> pd.DataFrame:
    """Wide month-end simple-return frame (index=month-end, cols=tickers) from the panel."""
    p = panel if tickers is None else panel[panel["ticker"].isin(tickers)]
    wide = (p.pivot_table(index="date", columns="ticker", values="adj_close")
              .sort_index()
              .resample("ME").last())
    return wide.pct_change()


def _benchmark_returns(panel: pd.DataFrame, benchmark: str):
    """Benchmark monthly returns. Uses the ticker if it's in the panel; otherwise an
    equal-weight universe proxy. Returns (series, label)."""
    uni = _monthly_returns(panel)
    if benchmark in uni.columns:
        return uni[benchmark].rename(benchmark), benchmark
    proxy = uni.mean(axis=1, skipna=True).rename(f"{benchmark}~EWproxy")
    return proxy, f"{benchmark} (equal-weight S&P-500 proxy — SPY not in offline panel)"


# --------------------------------------------------------------- weight caps
def _apply_caps(w: pd.Series, sectors: pd.Series,
                name_cap=NAME_CAP, sector_cap=SECTOR_CAP) -> pd.Series:
    """Project weights onto {sum=1, per-name ≤ name_cap, per-sector ≤ sector_cap} by
    alternating cap-and-redistribute. Converges in a handful of passes for sane caps."""
    w = w / w.sum()
    for _ in range(200):
        w = _cap_weights(w, name_cap)                      # per-name cap (sum stays 1)
        sec_tot = w.groupby(sectors).sum()
        over = sec_tot[sec_tot > sector_cap + 1e-9]
        if over.empty:
            break
        for sec, tot in over.items():                      # scale over-cap sectors down
            members = sectors[sectors == sec].index
            w[members] *= sector_cap / tot
        deficit = 1.0 - w.sum()                            # redistribute to under-cap names
        head = w[w < name_cap - 1e-9]
        if head.sum() <= 0:
            break
        w[head.index] += deficit * head / head.sum()
    return w / w.sum()


# --------------------------------------------------------------- selection
def _select(book: dict, top_n: int, betas: pd.Series, sectors: pd.Series):
    """Greedy top-score selection with a per-sector name cap, restricted to names that
    have a stable beta (≥ MIN_HISTORY obs, i.e. present in `betas`)."""
    ranked = book["holdings"].sort_values("model_score", ascending=False)
    ranked = ranked[ranked["ticker"].isin(betas.index)]    # only names with a stable beta
    picked, sec_count = [], {}
    for _, row in ranked.iterrows():
        tk = row["ticker"]
        sec = sectors.get(tk, "?")
        if sec_count.get(sec, 0) >= SECTOR_MAX_NAMES:
            continue
        picked.append(tk)
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(picked) == top_n:
            break
    if len(picked) < top_n:                                # relax the sector cap if starved
        for _, row in ranked.iterrows():
            if row["ticker"] not in picked:
                picked.append(row["ticker"])
            if len(picked) == top_n:
                break
    return picked


# --------------------------------------------------------------- beta targeting
def _hit_target_beta(w: pd.Series, betas: pd.Series, sectors: pd.Series, target: float):
    """Return (final_invested_weights, cash_weight, achieved_beta, note).

    w sums to 1 (fully invested). betas aligned to w.index. Long-only, never levers.
    """
    book_beta = float((w * betas.reindex(w.index)).sum())
    note = None
    if target <= book_beta:
        k = max(0.0, target / book_beta) if book_beta > 0 else 0.0
        final = w * k
        return final, float(1.0 - k), float(book_beta * k), note

    # target > book beta: tilt toward higher-beta names (no leverage available).
    b = betas.reindex(w.index).clip(lower=0)
    hi = _apply_caps(w * b, sectors)                       # caps-respecting high-beta book
    hi_beta = float((hi * betas.reindex(hi.index)).sum())
    if hi_beta < target - 1e-6:                            # even max tilt falls short → reset
        note = (f"Requested beta {target:.2f} exceeds the max achievable long-only beta "
                f"{hi_beta:.2f} for this book (no leverage in an educational sim). "
                f"Delivered the highest-beta long-only pie instead — expectation reset, "
                f"not faked.")
        return hi, 0.0, hi_beta, note
    lo, hiθ = 0.0, 1.0                                     # bisect blend θ: book→high beta
    for _ in range(60):
        mid = 0.5 * (lo + hiθ)
        blend = _apply_caps((1 - mid) * w + mid * hi, sectors)
        if float((blend * betas.reindex(blend.index)).sum()) < target:
            lo = mid
        else:
            hiθ = mid
    blend = _apply_caps((1 - hiθ) * w + hiθ * hi, sectors)
    return blend, 0.0, float((blend * betas.reindex(blend.index)).sum()), note


# --------------------------------------------------------------- figure
def _pie_chart(weights: pd.Series, cash_w: float, sectors: pd.Series, capital: float,
               target_beta: float, achieved: float, benchmark_label: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{tk} ({sectors.get(tk, '?')[:12]})" for tk in weights.index]
    sizes = list(weights.values)
    if cash_w > 1e-6:
        labels.append("CASH")
        sizes.append(cash_w)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(sizes, labels=labels, startangle=90, counterclock=False,
           autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
           textprops={"fontsize": 8})
    import textwrap
    disc = "\n".join(textwrap.wrap(DISCLAIMER, 84))
    ax.set_title(
        f"RankAlpha β-pie — ${capital:,.0f}  |  target β={target_beta:.2f}, "
        f"realised β={achieved:.2f}\nbenchmark: {benchmark_label}\n{disc}",
        fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------- API
def build_portfolio(capital, target_beta, benchmark="SPY", top_n=20,
                    date=None, panel_path="data/sp500_panel.parquet",
                    tickers_path="data/sp500_tickers.csv", make_figures=True, **score_kw):
    """Turn the frozen alpha ranking into a risk-matched, sector-diversified, LONG-ONLY
    pie whose realised beta ≈ `target_beta`. Educational simulation only.

    Steps: rank (frozen scores) → select top_n with a per-sector cap → inverse-vol weights
    with per-name & per-sector caps → hit target beta (cash sleeve or beta tilt) → score
    the pie with analytics.analyse() vs the benchmark → explain each slice → chart.
    """
    if capital <= 0:
        raise ValueError("capital must be positive")

    # 1. frozen scoring (cached; no refit) — gives holdings, SHAP reasons, sectors
    book = score_book(date=date, top_n=max(top_n * 3, 50), **score_kw)
    sectors = pd.Series({tk: ex.get("sector", "?") for tk, ex in book["explanations"].items()})

    # 2. monthly returns + benchmark from the committed panel
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    meta = pd.read_csv(tickers_path).set_index("ticker")
    sectors = sectors.reindex(sectors.index).fillna(meta["sector"] if "sector" in meta else "?")

    bench, bench_label = _benchmark_returns(panel, benchmark)
    cand = list(book["holdings"]["ticker"])
    stock_rets = _monthly_returns(panel, cand)

    # 3. per-stock beta on names with enough history (stable, common window)
    enough = [tk for tk in cand
              if tk in stock_rets and stock_rets[tk].dropna().shape[0] >= MIN_HISTORY]
    betas = pd.Series(
        {tk: _beta(stock_rets[tk], bench) for tk in enough}).dropna()

    # 4. select top_n with per-sector name cap, restricted to stable-beta names
    picked = _select(book, top_n, betas, sectors)
    picked = [tk for tk in picked if tk in betas.index]

    # 5. inverse-vol base weights with per-name + per-sector caps
    vol6 = book["holdings"].set_index("ticker").loc[picked, "vol_6m"]
    base = (1.0 / vol6)
    base = base / base.sum()
    sec_p = sectors.reindex(picked)
    capped = _apply_caps(base, sec_p)

    # 6. lock the scoring window FIRST, then compute the targeting betas on that SAME
    #    window — so the scorecard's realised beta equals Σwᵢβᵢ exactly (not just ≈).
    common = stock_rets[picked].dropna(how="any").tail(MAX_LOOKBACK)
    b_win = bench.reindex(common.index)
    beta_p = pd.Series({tk: _beta(common[tk], b_win) for tk in picked}).dropna()

    # 7. hit the target beta (cash sleeve or tilt; may reset expectation)
    weights, cash_w, achieved, note = _hit_target_beta(capped, beta_p, sec_p, target_beta)
    weights = weights.sort_values(ascending=False)

    # 8. score the realised static-weight pie vs the benchmark (β = Σwᵢβᵢ by construction)
    port = (common[list(weights.index)] * weights.reindex(common.columns)).sum(axis=1)
    scorecard = analyse(port, b_win, periods_per_year=PPY)
    scorecard["window"] = f"{port.index.min().date()} → {port.index.max().date()} " \
                          f"({len(port)} months, static current weights)"

    # 9. per-slice explanation: weight, sector, βᵢ (scoring-window), dominant factor reason
    slices = {}
    for tk in weights.index:
        ex = book["explanations"].get(tk, {})
        slices[tk] = {
            "weight": round(float(weights[tk]), 4),
            "dollars": round(float(weights[tk] * capital), 2),
            "sector": sectors.get(tk, "?"),
            "beta": round(float(beta_p[tk]), 3),
            "reason": ex.get("reasons", ["—"])[0],
        }

    dollar_alloc = {tk: round(float(w * capital), 2) for tk, w in weights.items()}
    dollar_alloc["CASH"] = round(cash_w * capital, 2)

    notes = [DISCLAIMER,
             "Realised beta is an in-sample match of TODAY's weights over history, "
             "not a forecast; out-of-sample beta drifts.",
             "Survivorship caveat: universe is today's S&P 500 members, not point-in-time.",
             f"alpha (annualised) reported alongside beta: {scorecard['alpha']:+.3f}"]
    if note:
        notes.insert(1, note)

    figures = {}
    if make_figures:
        tag = f"beta{int(round(target_beta * 100)):03d}"
        figures["pie"] = _pie_chart(weights, cash_w, sectors, capital, target_beta,
                                    achieved, bench_label, FIG_DIR / f"pie_{tag}.png")

    return {
        "as_of": book["as_of"],
        "capital": capital,
        "target_beta": target_beta,
        "achieved_beta": round(achieved, 3),
        "benchmark": bench_label,
        "weights": {tk: round(float(w), 4) for tk, w in weights.items()},
        "cash_weight": round(cash_w, 4),
        "dollar_allocations": dollar_alloc,
        "slices": slices,
        "scorecard": scorecard,
        "notes": notes,
        "figures": figures,
    }


def _print(p: dict):
    print("\n" + "=" * 74)
    print(f"RANKALPHA β-PIE — ${p['capital']:,.0f} as of {p['as_of']}  (LONG-ONLY, simulated)")
    print("=" * 74)
    print(f"benchmark: {p['benchmark']}")
    print(f"target β = {p['target_beta']:.2f}  ->  realised β = {p['achieved_beta']:.2f}"
          f"  |  cash sleeve {p['cash_weight']*100:.1f}%")
    s = p["scorecard"]
    print(f"scorecard [{s['window']}]:")
    print(f"  total ret {s['total_return']*100:+.1f}% | CAGR {s['cagr']*100:+.1f}% | "
          f"vol {s['volatility']*100:.1f}% | Sharpe {s['sharpe']:.2f} | "
          f"Sortino {s['sortino']:.2f}")
    print(f"  maxDD {s['max_drawdown']*100:.1f}% | alpha {s['alpha']:+.3f} | "
          f"beta {s['beta']:.3f}")
    print("-" * 74)
    print(f"{'Ticker':<7}{'wt':>7}{'$':>11}{'β':>7}  {'sector':<22} why")
    for tk, sl in p["slices"].items():
        print(f"{tk:<7}{sl['weight']*100:>6.1f}%{sl['dollars']:>11,.0f}{sl['beta']:>7.2f}  "
              f"{sl['sector'][:20]:<22} {sl['reason']}")
    print(f"{'CASH':<7}{p['cash_weight']*100:>6.1f}%{p['dollar_allocations']['CASH']:>11,.0f}")
    print("-" * 74)
    for n in p["notes"]:
        print("• " + n)
    print("figures:", ", ".join(p["figures"].values()))
    print("=" * 74 + "\n")


def _demo():
    for tb in (0.5, 1.0):
        _print(build_portfolio(10_000, target_beta=tb))


if __name__ == "__main__":
    _demo()
