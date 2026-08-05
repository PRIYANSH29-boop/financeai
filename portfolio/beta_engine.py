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


# --------------------------------------------------------------- sector map (Phase 20)
def _load_sector_map(tickers_path) -> pd.Series:
    """ticker -> sector, for the pie's sector caps. The S&P file carries `sector` inline;
    the wide universe file does not, so we resolve a committed companion `<stem>_sectors.csv`
    (produced by `scripts/map_sectors.py`). Returns an empty Series if neither exists — the
    caller then leaves sectors as '?'. This keeps the S&P path untouched (its inline column
    is used directly) while making the wide-universe caps active."""
    p = Path(tickers_path)
    try:
        meta = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=object)
    if "sector" in meta.columns:
        return meta.set_index("ticker")["sector"].astype(str)
    companion = p.with_name(f"{p.stem}_sectors.csv")
    if companion.exists():
        sec = pd.read_csv(companion)
        if {"ticker", "sector"} <= set(sec.columns):
            return sec.set_index("ticker")["sector"].astype(str)
    return pd.Series(dtype=object)


# --------------------------------------------------------------- weight caps
class CapsInfeasibleError(ValueError):
    """The requested book cannot satisfy the caps, so no weights are returned.

    #25 findings B-1/B-2/B-3: `_apply_caps` used to alternate for 200 passes and then
    `return w / w.sum()` unconditionally. When the pool was too small to absorb the deficit
    it returned a vector that VIOLATED the caps it exists to enforce — one name at 100%
    against an 8% cap — with no exception and no flag, and an all-zero book returned NaN
    weights. The caps are this product's central safety claim ("humility encoded as rules"),
    and a safety rule that fails open is worse than none: the shipped bundle was protected
    only because the exporter re-checked, so any other caller got the violation silently.

    Refusing is the honest answer. A pool that cannot be diversified has no diversified pie.
    """


def _caps_violations(w: pd.Series, sectors: pd.Series, name_cap: float, sector_cap: float,
                     tol: float = 1e-9) -> list[str]:
    """Cap breaches in `w`, checked exactly the way `_apply_caps` enforces them.

    Deliberately reuses `w.groupby(sectors)` rather than filling unmapped names into a '?'
    bucket: the verifier must test what the enforcer enforces, or it reports failures the
    loop was never trying to prevent. (Unmapped names escaping the sector cap is finding B-5,
    which is a separate instruction.)
    """
    bad = []
    if not np.isfinite(w.to_numpy(dtype="float64")).all():
        bad.append("non-finite weights")
        return bad                                         # everything below would be noise
    over_name = w[w > name_cap + tol]
    if len(over_name):
        worst = over_name.sort_values(ascending=False)
        bad.append(f"per-name cap {name_cap:.0%} breached by "
                   + ", ".join(f"{tk} {v:.1%}" for tk, v in worst.head(3).items()))
    sec_tot = w.groupby(sectors).sum()
    over_sec = sec_tot[sec_tot > sector_cap + tol]
    if len(over_sec):
        bad.append(f"per-sector cap {sector_cap:.0%} breached by "
                   + ", ".join(f"{s} {v:.1%}" for s, v in over_sec.items()))
    return bad


def _cap_names(w: pd.Series, cap: float) -> pd.Series:
    """Per-name cap with proportional redistribution, PRESERVING the incoming sum.

    `portfolio.engine._cap_weights` does the same thing but opens with `w = w / w.sum()`.
    That renormalise is correct in the vol engine (which always hands it a full book) and
    fatal here: the sector step deliberately leaves the book under-invested, so renormalising
    at the top of the next pass scaled the just-capped sector straight back over its cap. The
    two steps then traded weight for all 200 passes and the loop returned the oscillation.

    `portfolio/engine.py` is the shipped, frozen vol engine and is not ours to mutate (#15),
    so the sum-preserving variant lives here. With a full book in, this is `_cap_weights` out.
    """
    total = float(w.sum())
    if total <= 0:
        return w
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        if not under.any() or float(w[under].sum()) <= 0:
            break
        w[under] += excess * w[under] / float(w[under].sum())
    return w


def _apply_caps(w: pd.Series, sectors: pd.Series,
                name_cap=NAME_CAP, sector_cap=SECTOR_CAP) -> pd.Series:
    """Project weights onto {sum=1, per-name ≤ name_cap, per-sector ≤ sector_cap} by
    alternating cap-and-redistribute. Converges in a handful of passes for sane caps.

    Raises `CapsInfeasibleError` rather than returning weights that breach the caps (#25
    B-1/B-2/B-3). The commonest cause is a pool too small to hold the caps at all: with an
    8% name cap it takes ≥ 13 names to reach 100% invested, so a 3-name pool is arithmetically
    infeasible and there is no correct vector to return.
    """
    if len(w) == 0:
        raise CapsInfeasibleError("no names to weight: the candidate pool is empty")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        # B-3: `w / w.sum()` on an all-zero book yields NaN weights and no exception.
        raise CapsInfeasibleError(
            f"weights sum to {total!r} — an all-zero or non-finite book cannot be normalised. "
            f"This is a bug in the caller's scoring/sizing, not a portfolio.")
    if len(w) * name_cap < 1.0 - 1e-9:
        raise CapsInfeasibleError(
            f"{len(w)} name{'' if len(w) == 1 else 's'} cannot fill a book under a "
            f"{name_cap:.0%} per-name cap (max investable {len(w) * name_cap:.0%}). "
            f"Widen the pool or the cap.")
    # Joint capacity: each sector can hold at most min(sector_cap, n_names × name_cap), so a
    # pool concentrated in a few sectors can be arithmetically unable to be fully invested
    # however the weights are arranged. Checked up front because the alternative is a
    # confusing post-normalisation breach: the loop stops under-invested and the final
    # `w / w.sum()` then scales EVERY weight over its cap by the same shortfall factor.
    counts = sectors.reindex(w.index).fillna("?").value_counts()
    capacity = float(sum(min(sector_cap, n * name_cap) for n in counts))
    if capacity < 1.0 - 1e-9:
        top = ", ".join(f"{s} {n}" for s, n in counts.head(3).items())
        raise CapsInfeasibleError(
            f"{len(w)} names across {len(counts)} sectors can hold at most "
            f"{capacity:.1%} under a {name_cap:.0%} name cap and a {sector_cap:.0%} sector "
            f"cap, so no fully-invested book satisfies them (concentration: {top}). "
            f"Diversify the candidate pool or relax a cap — deliberately, not silently.")

    w = w / total
    for _ in range(200):
        w = _cap_names(w, name_cap)                        # per-name cap (sum preserved)
        sec_tot = w.groupby(sectors).sum()
        over = sec_tot[sec_tot > sector_cap + 1e-9]
        if over.empty:
            break
        for sec, tot in over.items():                      # scale over-cap sectors down
            members = sectors[sectors == sec].index
            w[members] *= sector_cap / tot
        deficit = 1.0 - w.sum()                            # redistribute to under-cap names
        # Redistribute into HEADROOM, not into current weight. The old rule spread the
        # deficit proportionally across every under-name-cap holding, including ones whose
        # SECTOR was already at its cap — which pushed that sector straight back over, and
        # the next pass scaled it down again. On a book with a heavy sector the two steps
        # simply traded weight back and forth for all 200 passes and the function returned
        # the oscillating state, over-cap, with no error. Capping the receipt at each name's
        # remaining room under BOTH caps makes the projection converge instead.
        sec_tot = w.groupby(sectors).sum()
        room_name = (name_cap - w).clip(lower=0.0)
        room_sector = sectors.reindex(w.index).map(
            lambda s: max(0.0, sector_cap - float(sec_tot.get(s, 0.0))))
        room = pd.concat([room_name, room_sector.astype(float)], axis=1).min(axis=1)
        if room.sum() <= 1e-15:
            break                                          # genuinely no room — verified below
        w = w + deficit * room / room.sum()

    w = w / w.sum()
    violations = _caps_violations(w, sectors, name_cap, sector_cap)
    if violations:
        # Fail CLOSED. Reaching here means the loop could not satisfy the constraints, which
        # for a sane pool means the pool itself is degenerate (e.g. every name in one sector).
        raise CapsInfeasibleError(
            f"cannot build a capped book from {len(w)} names: " + "; ".join(violations))
    return w


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
    if not np.isfinite(book_beta):
        # B-4: a NaN book beta used to flow through and be reported as a real number.
        raise CapsInfeasibleError(
            f"book beta is {book_beta!r} — at least one holding has no usable beta, so no "
            f"target can be hit or honestly reported.")
    if target <= book_beta:
        k = max(0.0, target / book_beta) if book_beta > 0 else 0.0
        final = w * k
        return final, float(1.0 - k), float(book_beta * k), note

    # target > book beta: tilt toward higher-beta names (no leverage available).
    b = betas.reindex(w.index).clip(lower=0)
    if float(b.sum()) <= 0:
        # B-4: every beta ≤ 0 and the target is above the book. `w * b` is all-zero, which
        # used to normalise to NaN weights and then report `achieved_beta = 0.0, cash = 0.0`
        # — a confident-looking number attached to an unusable book. There is nothing to tilt
        # toward, so say so.
        raise CapsInfeasibleError(
            f"target beta {target:.2f} is above the book's {book_beta:.2f} and no holding has "
            f"a positive beta to tilt toward — this book cannot reach any higher beta.")
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

    # 1. frozen scoring (cached; no refit) — gives holdings, SHAP reasons, sectors.
    #    Forward panel_path so score_book estimates book-vol on the SAME universe the pie is
    #    built on. Without this, score_book falls back to the default S&P panel and
    #    `_book_vol` KeyErrors on wide-universe names absent from it. tickers_path is left at
    #    the default on purpose: wide-universe sectors come back "?" and are resolved below
    #    from the committed companion map (score_book's own sector column stays S&P-only).
    book = score_book(date=date, top_n=max(top_n * 3, 50),
                      panel_path=panel_path, **score_kw)
    sectors = pd.Series({tk: ex.get("sector", "?") for tk, ex in book["explanations"].items()})

    # 2. monthly returns + benchmark from the committed panel
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])

    # Resolve sectors so the caps are active. score_book fills sectors from ITS own
    # tickers_path (the S&P file by default): wide-universe names come back '?', and — worse —
    # names that happen to also be in the S&P file get its GICS label ("Information
    # Technology") while the rest get the yfinance label ("Technology"), so one real sector
    # splits into two buckets and the ≤5-name cap is silently evaded. Fix: prefer the
    # tickers_path sector map for EVERY name — inline `sector` (S&P) or a committed
    # `<stem>_sectors.csv` companion (wide universe) — so one book uses one taxonomy. This is
    # a no-op for the S&P path (its map is the same inline column score_book already used).
    sector_map = _load_sector_map(tickers_path)
    if not sector_map.empty:
        sectors = pd.Series(
            [sector_map.get(tk) or sectors.get(tk, "?") for tk in sectors.index],
            index=sectors.index)
    sectors = sectors.fillna("?").replace({"nan": "?", "None": "?", "": "?"})

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
        # Unrounded weights. `weights` above is rounded to 4dp for printing, which makes a
        # 20-name book sum to ~0.9998 — invisible in a console table, but a donut chart with
        # a 0.02% gap is a defect. Consumers that need the sum to hold use these.
        "weights_exact": {tk: float(w) for tk, w in weights.items()},
        "cash_weight_exact": float(cash_w),
        "dollar_allocations": dollar_alloc,
        "slices": slices,
        "scorecard": scorecard,
        "notes": notes,
        "figures": figures,
        # Underlying monthly series, so downstream consumers (the #19 web-bundle exporter)
        # can derive equity/drawdown curves and window-scoped betas from the SAME numbers
        # the scorecard was computed on, instead of rebuilding the book and risking drift.
        # Additive only — nothing above changes.
        "monthly_returns": port,
        "benchmark_monthly_returns": b_win,
        "stock_monthly_returns": common[list(weights.index)],
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
