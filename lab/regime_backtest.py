"""
Regime-segmented backtest — Phase 21. Slice the EXISTING simulated history by market
weather and measure how each book behaves in calm / normal / stressed months.

⚠️ EDUCATIONAL SIMULATION ONLY. Nothing is trained here and nothing is predicted. This
module READS the frozen model's fixed-weight-today books and re-buckets their months by a
committed regime rule — no refit, no hand-picked windows, no look-ahead.

What it answers (from data we already have)
-------------------------------------------
1. Does the pie's cash sleeve + low-vol tilt earn its keep in stressed months?
2. How much does realised beta rise in stress? (calm-β vs stressed-β per book — the money
   number that turns the disclosed beta-drift caveat into a measured figure.)

The single-axis, single-construction choice (reviewer decision, #21)
--------------------------------------------------------------------
All four books are built the SAME way — today's fixed weights applied backward over the
identical panel window, measured against the identical equal-weight S&P-500 proxy:

  • Pie β0.50 / β0.75 / β1.00 : `beta_engine.build_portfolio` as shipped.
  • Momentum book             : RE-DERIVED here through the same weighting machinery from a
                                pure mom_12_1m score (top-N, inverse-vol, same caps, fully
                                invested). This is NOT the #14 walk-forward realized book;
                                it is a fixed-weight-today characterisation so it is a true
                                sibling of the pies. Named "Momentum (char.)" to keep the
                                distinction loud.

The regime calendar is computed ONCE on the beta-engine EW benchmark over that common
window; every book is sliced on the identical dates.

Hard limitations (stated up top, not buried)
--------------------------------------------
* NOT realized tracks. Every series here is an in-sample fixed-weight-today
  characterisation (the same caveat the #19 web bundle carries), NOT the 23-month realized
  paper-trade scorecard.
* The 2020 COVID crash is UNREACHABLE. The features data begins 2020-06-16, AFTER the
  Feb–Mar 2020 crash, and the picked names' common history only starts 2021-08. So the ONLY
  measurable stress episode is the 2022 bear. Stressed-regime stats rest on ~10 months and
  are DIRECTIONAL — no significance is claimed.
* Survivorship: the universe is today's S&P 500 members, not point-in-time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.beta_engine import (
    build_portfolio, _monthly_returns, _benchmark_returns, _apply_caps, _cap_names,
    MAX_LOOKBACK, MIN_HISTORY, PPY, NAME_CAP, SECTOR_CAP, SECTOR_MAX_NAMES,
)
from analytics.metrics import beta as _beta, max_drawdown

PANEL_PATH = Path("data/sp500_panel.parquet")
FEATURES_PATH = Path("data/sp500_features.parquet")
TICKERS_PATH = Path("data/sp500_tickers.csv")
REPORT_PATH = Path("figures/lab/regime_report.md")

PRESETS = [("Pie β0.50", 0.50), ("Pie β0.75", 0.75), ("Pie β1.00", 1.00)]
TOP_N = 20

# --- committed regime rule knobs (no hand-picked windows; documented, not tuned) --------
VOL_WINDOW = 3          # trailing months for realised-vol; drawdown leg does the heavy work
DD_STRESS = -0.10       # >10% below running peak  -> stressed
DD_CALM = -0.03         # within 3% of peak (with bottom-tercile vol) -> calm
STRESS_YEAR = 2022      # the one measurable stress episode; sanity gate asserts it fires


# ====================================================================== regime labelling
def classify_regimes(bench: pd.Series, vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    """Label each benchmark month calm / normal / stressed from a COMMITTED rule.

    stressed = drawdown < DD_STRESS  OR  trailing-vol in the top tercile
    calm     = trailing-vol in the bottom tercile  AND  drawdown > DD_CALM
    normal   = everything else   (stressed wins ties, then calm, then normal)

    Returns a frame indexed like `bench` with drawdown, roll_vol, vol_tercile, regime.
    """
    b = bench.dropna().astype("float64")
    eq = (1.0 + b).cumprod()
    dd = eq / eq.cummax() - 1.0
    # annualised realised vol over the trailing `vol_window` months (population std)
    roll_vol = b.rolling(vol_window, min_periods=2).std(ddof=0) * np.sqrt(PPY)

    valid = roll_vol.dropna()
    q_lo, q_hi = valid.quantile(1 / 3), valid.quantile(2 / 3)

    def tercile(v):
        if np.isnan(v):
            return "n/a"          # early months with no vol yet
        if v >= q_hi:
            return "top"
        if v <= q_lo:
            return "bottom"
        return "mid"

    terc = roll_vol.map(tercile)

    def label(d, t):
        if d < DD_STRESS or t == "top":
            return "stressed"
        if t == "bottom" and d > DD_CALM:
            return "calm"
        return "normal"

    def trigger(d, t):
        """Why a stressed month is stressed — so the drawdown core (the 2022 bear) is not
        conflated with the always-⅓ top-vol-tercile months. Empty for non-stressed."""
        dd_hit, vol_hit = d < DD_STRESS, t == "top"
        if dd_hit and vol_hit:
            return "dd+vol"
        if dd_hit:
            return "dd"
        if vol_hit:
            return "vol"
        return ""

    regime = pd.Series([label(dd[i], terc[i]) for i in b.index], index=b.index)
    trig = pd.Series([trigger(dd[i], terc[i]) for i in b.index], index=b.index)
    return pd.DataFrame({"drawdown": dd, "roll_vol": roll_vol,
                         "vol_tercile": terc, "regime": regime, "trigger": trig})


# ================================================================= re-derived momentum book
def _legacy_uncapped_weights(base: pd.Series, sectors: pd.Series) -> pd.Series:
    """⚠️ RETRACTED CONSTRUCTION — reproduces the pre-#28 `_apply_caps` exactly, breaches and
    all. It exists for ONE purpose: to regenerate the momentum column that was published in
    this report before #28, so the correction can be shown side by side instead of silently
    overwritten. **Never use it to build a book anyone acts on.**

    The pre-#28 projection alternated cap-and-redistribute for 200 passes and then returned
    `w / w.sum()` unconditionally. On a pool that cannot satisfy the caps it therefore
    returned weights that VIOLATED them — here, Information Technology at 44.0% against a
    30% cap — with no exception and no flag.
    """
    w = base / base.sum()
    for _ in range(200):
        w = _cap_names(w, NAME_CAP)
        sec_tot = w.groupby(sectors).sum()
        over = sec_tot[sec_tot > SECTOR_CAP + 1e-9]
        if over.empty:
            break
        for sec, tot in over.items():
            members = sectors[sectors == sec].index
            w[members] *= SECTOR_CAP / tot
        deficit = 1.0 - w.sum()
        head = w[w < NAME_CAP - 1e-9]
        if head.sum() <= 0:
            break
        w[head.index] += deficit * head / head.sum()
    return w / w.sum()                       # <- the unconditional renormalise: the bug


def momentum_book(panel: pd.DataFrame, top_n: int = TOP_N,
                  features_path: Path = FEATURES_PATH,
                  tickers_path: Path = TICKERS_PATH,
                  sector_max_names: int = SECTOR_MAX_NAMES,
                  legacy_uncapped: bool = False):
    """A pure-momentum book built through the SAME fixed-weight machinery as the pies, so it
    is a true sibling (NOT the #14 realized book): latest-date top-N by mom_12_1m, inverse
    vol_6m weights with the same per-name/per-sector caps, fully invested, measured over the
    same common window. Returns (port_series, weights).

    #30 Part A — selection now applies the pies' own per-sector NAME limit
    (`SECTOR_MAX_NAMES`), which is what makes the book cap-FEASIBLE by construction. Before
    this, selection took the raw momentum top-20 with no sector limit, which on the committed
    panel is 13 of 20 names in Information Technology: a pool that can hold at most 86% under
    an 8% name cap and a 30% sector cap, so no fully-invested book satisfies them. The old
    code returned it anyway with Information Technology at 44%. Since #28 the caps fail
    closed, so that path now raises rather than lying — this is the repair.

    `legacy_uncapped=True` reproduces the retracted pre-#28 book for the correction table.
    """
    feats = pd.read_parquet(features_path)
    feats["date"] = pd.to_datetime(feats["date"])
    as_of = feats["date"].max()
    day = feats[feats["date"] == as_of].dropna(subset=["mom_12_1m", "vol_6m"])

    meta = pd.read_csv(tickers_path).set_index("ticker")
    sectors = meta["sector"] if "sector" in meta else pd.Series(dtype=object)

    stock_rets = _monthly_returns(panel, list(day["ticker"]))
    # restrict to names with a stable, common history (same MIN_HISTORY floor as the pies)
    eligible = [tk for tk in day["ticker"]
                if tk in stock_rets and stock_rets[tk].dropna().shape[0] >= MIN_HISTORY]
    day = day[day["ticker"].isin(eligible)].sort_values("mom_12_1m", ascending=False)

    if legacy_uncapped:
        picked = list(day["ticker"].head(top_n))          # raw top-N, no sector limit
    else:
        # Mirror `beta_engine._select`: greedy by score, at most `sector_max_names` per
        # sector. Same rule as the pies, so the sibling really is a sibling.
        picked, sec_count = [], {}
        for tk in day["ticker"]:
            sec = sectors.get(tk, "?")
            if sec_count.get(sec, 0) >= sector_max_names:
                continue
            picked.append(tk)
            sec_count[sec] = sec_count.get(sec, 0) + 1
            if len(picked) == top_n:
                break

    vol6 = day.set_index("ticker").loc[picked, "vol_6m"]
    base = (1.0 / vol6)
    base = base / base.sum()
    sec_p = sectors.reindex(picked).fillna("?")
    project = _legacy_uncapped_weights if legacy_uncapped else _apply_caps
    weights = project(base, sec_p).sort_values(ascending=False)

    common = stock_rets[picked].dropna(how="any").tail(MAX_LOOKBACK)
    port = (common[list(weights.index)] * weights.reindex(common.columns)).sum(axis=1)
    return port, weights


# ============================================================================ regime stats
def _hit_rate(r: pd.Series) -> float:
    return float((r > 0).mean()) if len(r) else float("nan")


def regime_stats(port: pd.Series, bench: pd.Series, labels: pd.Series) -> dict:
    """Per-regime stats for one book on the regime calendar. All analyser conventions:
    population std (ddof=0), monthly, PPY=12. Beta is per regime = cov/var on THOSE months."""
    # align all three to a common monthly index
    idx = port.index.intersection(bench.index).intersection(labels.index)
    port, bench, labels = port.reindex(idx), bench.reindex(idx), labels.reindex(idx)

    out = {}
    for reg in ("calm", "normal", "stressed", "all"):
        mask = labels.notna() if reg == "all" else (labels == reg)
        r, b = port[mask], bench[mask]
        n = int(len(r))
        if n == 0:
            out[reg] = {"n": 0, "mean": None, "vol": None, "maxdd": None,
                        "beta": None, "hit": None}
            continue
        eq = (1.0 + r).cumprod().to_numpy()
        mdd, _ = max_drawdown(eq)
        out[reg] = {
            "n": n,
            "mean": float(r.mean()),
            "vol": float(r.std(ddof=0) * np.sqrt(PPY)),
            "maxdd": float(mdd),
            "beta": float(_beta(r, b)) if n >= 2 and b.var(ddof=0) > 0 else None,
            "hit": _hit_rate(r),
        }
    return out


def panel_sectors(tickers_path: Path = TICKERS_PATH) -> pd.Series:
    """ticker -> sector for the S&P file, or an empty Series if the column is absent."""
    meta = pd.read_csv(tickers_path).set_index("ticker")
    return meta["sector"] if "sector" in meta else pd.Series(dtype=object)


def _sector_profile(w: pd.Series, sectors: pd.Series) -> dict:
    """The cap evidence for one book: how concentrated it is and whether it breaches.

    Reported for BOTH momentum books so the correction is arithmetic the reader can check,
    not an assertion they have to take on trust.
    """
    sec = sectors.reindex(w.index).fillna("?")
    tot = w.groupby(sec).sum().sort_values(ascending=False)
    counts = sec.value_counts()
    capacity = float(sum(min(SECTOR_CAP, n * NAME_CAP) for n in counts))
    return {
        "n_names": int(len(w)),
        "n_sectors": int(len(counts)),
        "max_name": float(w.max()),
        "top_sector": str(tot.index[0]),
        "top_sector_weight": float(tot.iloc[0]),
        "top_sector_names": int(counts.get(tot.index[0], 0)),
        "capacity": capacity,
        "breaches_name_cap": bool(w.max() > NAME_CAP + 1e-9),
        "breaches_sector_cap": bool(tot.iloc[0] > SECTOR_CAP + 1e-9),
    }


# ================================================================================= assembly
def run(make_report: bool = True) -> dict:
    """Build all books on one axis, classify regimes, compute per-regime stats, and (opt.)
    write figures/lab/regime_report.md. Returns the structured result dict."""
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    # 1. the three pies (each carries its own port + benchmark on the common window)
    books, bench = {}, None
    for label, tb in PRESETS:
        p = build_portfolio(10_000, target_beta=tb, top_n=TOP_N, make_figures=False)
        books[label] = p["monthly_returns"]
        bench = p["benchmark_monthly_returns"] if bench is None else bench

    # 2. the re-derived momentum sibling — capped selection (#30 Part A)
    mom_port, mom_w = momentum_book(panel, top_n=TOP_N)
    books["Momentum (char.)"] = mom_port

    # 2b. the RETRACTED pre-#28 book, rebuilt only so the correction is visible in the
    #     report rather than silently replacing the published numbers.
    legacy_port, legacy_w = momentum_book(panel, top_n=TOP_N, legacy_uncapped=True)
    books["Momentum (uncapped — retracted)"] = legacy_port

    # 3. align everything to ONE common monthly index (the regime axis), incl. benchmark
    common_idx = bench.index
    for s in books.values():
        common_idx = common_idx.intersection(s.index)
    bench = bench.reindex(common_idx)
    books = {k: v.reindex(common_idx) for k, v in books.items()}
    books["EW benchmark"] = bench  # benchmark sliced too (β=1.0 by definition)

    # 4. committed regime calendar on the benchmark
    reg = classify_regimes(bench)
    labels = reg["regime"].reindex(common_idx)
    counts = labels.value_counts().to_dict()
    stressed_mask = labels == "stressed"
    stressed_months = [d.date().isoformat() for d in labels.index[stressed_mask]]
    trig = reg["trigger"].reindex(common_idx)
    stress_triggers = trig[stressed_mask].value_counts().to_dict()
    # drawdown-driven months = the genuine bear core (>10% off peak)
    dd_driven = [d.date().isoformat()
                 for d in trig.index[stressed_mask & trig.isin(["dd", "dd+vol"])]]

    # 5. SANITY GATE — the one measurable stress episode (2022 bear) MUST fire.
    #    (The 2020 crash is unreachable: features start 2020-06, post-crash — see module doc.)
    stressed_years = {int(m[:4]) for m in stressed_months}
    if STRESS_YEAR not in stressed_years:
        raise AssertionError(
            f"Sanity gate FAILED: the {STRESS_YEAR} bear did not land in 'stressed'. "
            f"Stressed months: {stressed_months}. The committed rule is wrong — stop and "
            f"report, do not paper over it.")

    # 6. per-book, per-regime stats
    stats = {name: regime_stats(series, bench, labels) for name, series in books.items()}

    # 6b. #21 rider — the stressed-CORE: only the >10%-drawdown months (the genuine 2022
    #     bear, n=6), reported alongside the full stressed bucket. Labelling only these as
    #     'stressed' makes regime_stats return the core stats under its 'stressed' key.
    core_mask = stressed_mask & trig.isin(["dd", "dd+vol"])
    core_labels = pd.Series(np.where(core_mask, "stressed", "excl"), index=common_idx)
    core_stats = {name: regime_stats(series, bench, core_labels)["stressed"]
                  for name, series in books.items()}

    result = {
        "window": f"{common_idx.min().date()} → {common_idx.max().date()}",
        "n_months": int(len(common_idx)),
        "regime_counts": counts,
        "stressed_months": stressed_months,
        "stress_triggers": stress_triggers,
        "drawdown_driven_months": dd_driven,
        "regime_frame": reg,
        "stats": stats,
        "core_stats": core_stats,
        "momentum_weights": {
            "capped": _sector_profile(mom_w, panel_sectors(TICKERS_PATH)),
            "retracted": _sector_profile(legacy_w, panel_sectors(TICKERS_PATH)),
        },
    }
    if make_report:
        result["report_path"] = str(write_report(result))
    return result


# ================================================================================== report
def _fmt_pct(x):
    return "—" if x is None else f"{x * 100:+.2f}%"


def _fmt_dd(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def _fmt_beta(x):
    return "—" if x is None else f"{x:.3f}"


def _fmt_ratio(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def write_report(result: dict) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = result["regime_counts"]
    L = []
    L.append("# RankAlpha — regime-segmented backtest (Phase 21)\n")
    L.append("*Auto-generated by `lab/regime_backtest.py`. Educational SIMULATION only — "
             "no refit, no prediction, no network. Regenerated from committed data.*\n")
    L.append(f"**Window:** {result['window']} ({result['n_months']} months). "
             "All four books are today's fixed weights applied backward over the SAME "
             "window, measured against the SAME equal-weight S&P-500 proxy. The regime "
             "calendar is computed once on that benchmark.\n")

    L.append("## ⚠️ Read this before the numbers\n")
    L.append("* **Not realized tracks.** Every series is an in-sample fixed-weight-today "
             "*characterisation* (same caveat the #19 web bundle carries), NOT the 23-month "
             "realized paper-trade scorecard.\n")
    L.append("* **The 2020 crash is unreachable.** Features begin **2020-06-16**, *after* "
             "the Feb–Mar 2020 crash, and the picked names' common history only starts "
             "2021-08. The **only** measurable stress episode is the **2022 bear**. "
             "Stressed stats rest on a handful of months and are **DIRECTIONAL — no "
             "significance is claimed.**\n")
    # Computed, not typed. This caveat used to hardcode "+9.51%/mo" and "1.5x market in
    # stress" — both of which described the book #30 retracted, so a repair that left the
    # prose alone would have published a corrected table under a stale warning.
    _m = result["stats"].get("Momentum (char.)", {})
    _calm_mean = (_m.get("calm") or {}).get("mean")
    _calm_b = (_m.get("calm") or {}).get("beta")
    _str_b = (_m.get("stressed") or {}).get("beta")
    L.append("* **\"Momentum (char.)\"** is re-derived through the pies' fixed-weight "
             "machinery — a sibling of the pies, **not** the #14 walk-forward realized "
             "book. Its numbers are valid **only as a beta-drift illustration** "
             f"(calm β {_fmt_beta(_calm_b)} → {_fmt_beta(_str_b)} in stress). They are "
             f"**never quotable as an achievable return**: the calm {_fmt_pct(_calm_mean)}/mo "
             "figure is a fixed-weights-applied-backward + survivorship artifact, not "
             "something any investor could have earned.\n")
    L.append("* **Survivorship:** today's S&P 500 members, not point-in-time.\n")

    L.append("## Regime calendar (committed rule)\n")
    L.append("Rule: `stressed = drawdown < −10% OR trailing-3mo vol in top tercile` · "
             "`calm = bottom-tercile vol AND drawdown > −3%` · `normal = else`.\n")
    L.append(f"**Month counts:** calm {counts.get('calm', 0)} · "
             f"normal {counts.get('normal', 0)} · stressed {counts.get('stressed', 0)}\n")
    L.append(f"**Stressed months ({len(result['stressed_months'])}):** "
             + ", ".join(result["stressed_months"]) + "\n")
    trg = result["stress_triggers"]
    dd_driven = result["drawdown_driven_months"]
    L.append(f"**Why stressed:** {trg.get('dd', 0)} drawdown-only · "
             f"{trg.get('dd+vol', 0)} drawdown+high-vol · {trg.get('vol', 0)} high-vol-only. "
             "The **top-tercile-vol leg is always ~⅓ of months by construction**, so it "
             "pulls in moderate-vol months well beyond the bear. The genuine drawdown core "
             f"(>10% off peak — the **2022 bear**) is **{len(dd_driven)} months**: "
             + ", ".join(dd_driven) + ".\n")
    L.append("_(Sanity gate: the 2022 bear lands in 'stressed' ✅. The 2020 crash is absent "
             "by data availability, not by rule error.)_\n")

    # --- #30 Part A: the correction, stated before any number that it changes ------------
    mw = result.get("momentum_weights")
    if mw:
        cap, ret = mw["capped"], mw["retracted"]
        L.append("## ⛔ Correction (#30) — the momentum column was rebuilt\n")
        L.append("The momentum sibling published here before 2026-08-05 **violated the very "
                 "caps this report says it was built under**. Selection took the raw "
                 f"momentum top-{TOP_N} with no per-sector name limit, which on the committed "
                 f"panel is **{ret['top_sector_names']} of {ret['n_names']} names in "
                 f"{ret['top_sector']}**. A pool that concentrated can hold at most "
                 f"**{ret['capacity']:.1%}** under an {NAME_CAP:.0%} name cap and a "
                 f"{SECTOR_CAP:.0%} sector cap, so **no fully-invested book satisfies "
                 "them** — the old projection returned one anyway, over-cap and unflagged.\n")
        L.append("Selection now applies the pies' own per-sector name limit "
                 f"(≤{SECTOR_MAX_NAMES}/sector), the same rule `beta_engine._select` uses, "
                 "which makes the book cap-feasible by construction. **Both columns are "
                 "shown below.** The retracted one is kept because a correction that erases "
                 "the thing it corrects is not a correction.\n")
        L.append("| | Retracted (uncapped selection) | Repaired (#30) |")
        L.append("|---|---|---|")
        L.append(f"| Names | {ret['n_names']} | {cap['n_names']} |")
        L.append(f"| Sectors | {ret['n_sectors']} | {cap['n_sectors']} |")
        L.append(f"| Largest sector | **{ret['top_sector']} "
                 f"{ret['top_sector_weight']:.1%}** ({ret['top_sector_names']} names) | "
                 f"{cap['top_sector']} {cap['top_sector_weight']:.1%} "
                 f"({cap['top_sector_names']} names) |")
        L.append(f"| Largest single name | {ret['max_name']:.1%} | {cap['max_name']:.1%} |")
        L.append(f"| Max investable under both caps | {ret['capacity']:.1%} | "
                 f"{cap['capacity']:.1%} |")
        L.append(f"| Breaches the {SECTOR_CAP:.0%} sector cap | "
                 f"**{'YES' if ret['breaches_sector_cap'] else 'no'}** | "
                 f"{'YES' if cap['breaches_sector_cap'] else 'no'} |")
        L.append(f"| Breaches the {NAME_CAP:.0%} name cap | "
                 f"**{'YES' if ret['breaches_name_cap'] else 'no'}** | "
                 f"{'YES' if cap['breaches_name_cap'] else 'no'} |")
        L.append("")
        L.append("**The pie rows are unaffected and were not re-derived** — #28 verified the "
                 "shipped pies bit-identical across the whole beta grid (max |Δw| = 0.0), "
                 "and `lab/tests/test_regime_backtest.py` asserts the pie books stay "
                 "unchanged by this repair. Only the momentum column moves.\n")

    order = ["Pie β0.50", "Pie β0.75", "Pie β1.00",
             "Momentum (char.)", "Momentum (uncapped — retracted)", "EW benchmark"]
    for reg in ("calm", "normal", "stressed"):
        L.append(f"## Regime: {reg.upper()}\n")
        L.append("| Book | n | Mean/mo | Vol (ann.) | MaxDD | Realised β | Hit rate |")
        L.append("|---|---|---|---|---|---|---|")
        for name in order:
            s = result["stats"][name][reg]
            L.append(f"| {name} | {s['n']} | {_fmt_pct(s['mean'])} | "
                     f"{_fmt_pct(s['vol'])} | {_fmt_dd(s['maxdd'])} | "
                     f"{_fmt_beta(s['beta'])} | {_fmt_ratio(s['hit'])} |")
        L.append("")

        # #21 rider — after the full stressed bucket, break out the stressed-CORE: only the
        # >10%-drawdown months (the genuine 2022 bear), so the reader sees the bear alone,
        # not diluted by the always-⅓ top-vol-tercile months.
        if reg == "stressed":
            core_n = result["core_stats"]["EW benchmark"]["n"]
            L.append(f"### Stressed-CORE — the {core_n} >10%-drawdown months only "
                     "(2022 bear, no vol-tercile dilution)\n")
            L.append(f"Months: {', '.join(result['drawdown_driven_months'])}. Even more "
                     "directional than the full bucket — read as illustration, not evidence.\n")
            L.append("| Book | n | Mean/mo | Vol (ann.) | MaxDD | Realised β | Hit rate |")
            L.append("|---|---|---|---|---|---|---|")
            for name in order:
                s = result["core_stats"][name]
                L.append(f"| {name} | {s['n']} | {_fmt_pct(s['mean'])} | "
                         f"{_fmt_pct(s['vol'])} | {_fmt_dd(s['maxdd'])} | "
                         f"{_fmt_beta(s['beta'])} | {_fmt_ratio(s['hit'])} |")
            L.append("")

    # calm-β vs stressed-β — the money table
    L.append("## Calm-β vs stressed-β — the beta-drift number\n")
    L.append("| Book | Calm β | Stressed β | Δ (stress − calm) |")
    L.append("|---|---|---|---|")
    for name in order:
        cb = result["stats"][name]["calm"]["beta"]
        sb = result["stats"][name]["stressed"]["beta"]
        d = None if (cb is None or sb is None) else sb - cb
        L.append(f"| {name} | {_fmt_beta(cb)} | {_fmt_beta(sb)} | {_fmt_beta(d)} |")
    L.append("")

    REPORT_PATH.write_text("\n".join(L))
    return REPORT_PATH


def main():
    res = run(make_report=True)
    print(f"window {res['window']} ({res['n_months']} mo) | "
          f"regimes {res['regime_counts']}")
    print("stressed:", ", ".join(res["stressed_months"]))
    print("report ->", res.get("report_path"))


if __name__ == "__main__":
    main()
