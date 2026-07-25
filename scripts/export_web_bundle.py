#!/usr/bin/env python3
"""
Web bundle exporter — Phase 19, deliverable A.

Precomputes every number the static frontend will ever display, so the web app can be a
pure `output: 'export'` Next.js build with no backend, no API keys, and no runtime data
fetches. **No number may be typed into the frontend** — if it is not in this bundle, the UI
does not get to show it.

    make web-bundle
    python scripts/export_web_bundle.py --out web/public/bundle

What is emitted
---------------
    web/public/bundle/
        index.json          global: beta grid, presets, max achievable beta, benchmark
                            series, guardrails, caveats, schema version
        beta/b000.json      one file per beta on the grid (0.00 → 1.85 step 0.05)
        beta/b050.json      …presets 0.50 / 0.75 / 1.00 are grid points, not special cases
        …

Per-beta files are separate so the static site fetches ~15 KB when the slider moves, rather
than a single multi-megabyte blob on first paint.

Design rules this file enforces
-------------------------------
* **The engine is the only source of truth.** Every figure comes from
  `portfolio.beta_engine.build_portfolio` on the SHIPPED S&P 500 frozen model — never the
  Phase 16 mid+large model, which was explicitly rejected as the shipped one. Portfolio
  construction is not reimplemented here; the exporter reads the engine's own monthly
  series so curves cannot drift from the scorecard.
* **Max achievable beta is asked, not assumed.** The engine caps impossible targets and
  resets the expectation; we request an absurd target and record what it actually achieved.
  Nothing is hardcoded — if the engine changes, the bundle changes with it.
* **Weights are capital-independent.** Capital is a client-side multiplier, so no dollar
  amounts are exported at all. This is what lets one bundle serve any capital input.
* **The beta-drift disclosure is mandatory** (#19): each pie exports its realised beta
  measured *inside its own max-drawdown window*, which is where a target-beta promise is
  most likely to have failed the user.

⚠️ EDUCATIONAL SIMULATION. Historical characterisation of today's weights, never a
forecast. Survivorship-biased universe. No projections of any kind are exported — there is
deliberately no field a frontend could misread as a forward return.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analytics.metrics import (  # noqa: E402
    beta as _beta, equity_curve, volatility, sharpe, sortino, max_drawdown, total_return,
)
from portfolio.beta_engine import (  # noqa: E402
    build_portfolio, _monthly_returns, MAX_LOOKBACK, MIN_HISTORY,
    NAME_CAP, SECTOR_CAP, SECTOR_MAX_NAMES,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("web_bundle")

SCHEMA_VERSION = 1
OUT_DIR = Path("web/public/bundle")

BETA_MIN, BETA_MAX, BETA_STEP = 0.00, 1.85, 0.05
PRESETS = [("Defensive", 0.50), ("Balanced", 0.75), ("Market", 1.00)]

# Requesting this makes the engine tilt as far as long-only allows, then cap and reset the
# expectation. Whatever it reports back IS the max achievable beta. Never hardcode it.
ABSURD_BETA = 5.0

TICKERS_CSV = Path("data/sp500_tickers.csv")

# Phase 23 — Explore (wide universe) + Basket (S&P scored names) data sources. All gitignored
# build inputs; the bundle they produce IS committed, so the frontend never needs them.
SP500_PANEL = Path("data/sp500_panel.parquet")
MIDLARGE_PANEL = Path("data/midlarge_panel.parquet")
MIDLARGE_UNIVERSE = Path("data/universe_midlarge.csv")
MIDLARGE_SECTORS = Path("data/universe_midlarge_sectors.csv")
CAP_LARGE_USD = 10_000_000_000       # ≥ $10B = large-cap, $2–10B = mid-cap (decided #23)
MOVERS_N = 10                        # winners/losers card size
BASKET_MIN_HISTORY = 24              # a scored name needs ≥24 monthly obs to be basket-usable

# Phase 24 — statistical sanity band for per-name displayed stats.
#
# A monthly beta of 364 or an annualised vol of 160% is not a risk measurement, it is an
# artifact: a reorganisation spliced into one adjusted-close series (CHRD's Nov-2020
# +30,991% month is the pre/post-bankruptcy Whiting splice), or a few months of history
# pretending to be an estimate. The band does NOT change the numbers we display — a
# winsorised beta shown as if measured is exactly the silent-wrong-answer this project
# exists to avoid. It flags them, keeps them out of the default risk orderings, and bars
# them from the basket, where a bogus beta would drive a scorecard the user acts on.
BETA_ABS_MAX = 3.0
ANN_VOL_MAX = 1.0

# Why a name is not basket-eligible / why its stats are flagged. Exported verbatim so the
# UI never has to invent an explanation for a name it is refusing to offer.
FLAG_REASONS = {
    "no_history": "no return history in the committed panel",
    "insufficient_history": f"fewer than {BASKET_MIN_HISTORY} continuous recent monthly "
                            f"observations",
    "discontinuous": "gaps inside its own price history",
    "stale_history": "no return in the most recent month — not currently tradable",
    "beta_out_of_band": f"beta outside ±{BETA_ABS_MAX:.0f} — unreliable estimate",
    "vol_out_of_band": f"annualised volatility above {ANN_VOL_MAX:.0%} — unreliable estimate",
    "beta_unavailable": "beta could not be estimated",
    "vol_unavailable": "volatility could not be estimated",
    "not_in_model_universe": "not in the S&P-500 set the frozen model ranks",
}

# Shown verbatim in the UI's "How honest is this?" box (#19 requires these exact points).
CAVEATS = [
    "23-month simulated track — too short to be statistically significant",
    "survivorship-biased universe",
    "target beta estimated from calm markets; real beta rises in crashes",
    "educational simulation, not investment advice",
]

HOW_IT_WORKS = [
    "Rank every S&P 500 stock with the frozen model score.",
    "Apply the guardrails — at most 8% per stock, 30% per sector, 5 stocks per sector.",
    "Size what is left by inverse volatility, so calmer stocks carry more weight.",
    "Add a cash sleeve until the pie's beta matches your target. Cash has a beta of zero.",
]


def beta_grid() -> list[float]:
    """The slider's discrete stops. Built with integer arithmetic so 0.05 steps land on
    exact 2dp values and the filename key always round-trips."""
    n = int(round((BETA_MAX - BETA_MIN) / BETA_STEP))
    return [round(BETA_MIN + i * BETA_STEP, 2) for i in range(n + 1)]


def beta_key(b: float) -> str:
    """0.75 -> 'b075'. Stable, sortable, filesystem-safe, and unambiguous at 2dp."""
    return f"b{int(round(b * 100)):03d}"


# ------------------------------------------------------------------ series helpers
def _series(index, values) -> list[dict]:
    """Month-end series as [{d: 'YYYY-MM-DD', v: float}] — compact and chart-ready."""
    out = []
    for d, v in zip(index, values):
        f = float(v)
        if np.isfinite(f):
            out.append({"d": pd.Timestamp(d).strftime("%Y-%m-%d"), "v": round(f, 6)})
    return out


def _drawdown(returns: pd.Series) -> np.ndarray:
    eq = equity_curve(returns.to_numpy(dtype="float64"))
    return eq / np.maximum.accumulate(eq) - 1.0


def _drawdown_window(returns: pd.Series) -> tuple[int, int]:
    """(start, end) positional index of the deepest drawdown: peak before it, trough at it."""
    dd = _drawdown(returns)
    if len(dd) == 0:
        return 0, 0
    trough = int(np.argmin(dd))
    eq = equity_curve(returns.to_numpy(dtype="float64"))
    peak = int(np.argmax(eq[:trough + 1])) if trough > 0 else 0
    return peak, trough


def beta_in_drawdown(port: pd.Series, bench: pd.Series) -> dict:
    """Realised beta measured INSIDE the worst-drawdown window — the mandatory #19 disclosure.

    A target-beta pie is sold on a beta estimated over the whole (mostly calm) history. The
    number that actually matters to someone holding it is what the beta did while the pie
    was falling. Exporting it is not optional: a UI that shows only the headline beta is
    quietly making the promise this project exists to avoid making.
    """
    start, end = _drawdown_window(port)
    seg_p, seg_b = port.iloc[start:end + 1], bench.iloc[start:end + 1]
    n = int(len(seg_p))
    if n < 3:            # too few points for a meaningful covariance — say so, don't guess
        return {"beta": None, "n_months": n, "start": None, "end": None,
                "note": "drawdown window too short to estimate a beta"}
    return {
        "beta": round(float(_beta(seg_p, seg_b)), 3),
        "n_months": n,
        "start": pd.Timestamp(seg_p.index[0]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(seg_p.index[-1]).strftime("%Y-%m-%d"),
        "note": "Realised beta measured inside the worst drawdown, where a calm-market "
                "beta estimate is most likely to have understated the risk.",
    }


def drift_after_one_period(weights: pd.Series, stock_rets: pd.DataFrame,
                           cash_weight: float) -> dict:
    """Weights after ONE rebalance period of committed price history, with per-holding delta.

    This answers the wireframe's "what happens after you invest?" honestly: you buy the
    target pie, prices move, and by the next rebalance your weights are no longer the ones
    you chose. Winners grow past their cap, losers shrink. Cash does not move (return 0),
    so its share rises whenever the book falls.

    Deltas are in **percentage points of weight** — not returns, and not losses. The UI
    renders them in a neutral colour for exactly that reason.
    """
    if stock_rets is None or stock_rets.empty:
        return {"available": False, "reason": "no monthly return history available"}
    last = stock_rets.iloc[-1].reindex(weights.index)
    if last.isna().all():
        return {"available": False, "reason": "last period has no usable returns"}
    last = last.fillna(0.0)

    grown = weights * (1.0 + last)
    total = float(grown.sum()) + float(cash_weight)      # cash earns 0 and still counts
    if total <= 0:
        return {"available": False, "reason": "degenerate total weight"}
    drifted = grown / total
    drifted_cash = float(cash_weight) / total

    rows = []
    for tk in weights.index:
        # Derive the delta from the ROUNDED pair, so `target + delta == drifted` holds
        # exactly on the numbers the UI renders. Deriving it from full precision and then
        # rounding independently lets the three displayed figures disagree in the last digit.
        tw, dw = round(float(weights[tk]), 6), round(float(drifted[tk]), 6)
        rows.append({"ticker": tk, "target_weight": tw, "drifted_weight": dw,
                     "delta_pp": round((dw - tw) * 100.0, 4)})
    rows.sort(key=lambda r: -abs(r["delta_pp"]))
    return {
        "available": True,
        "period_end": pd.Timestamp(stock_rets.index[-1]).strftime("%Y-%m-%d"),
        "holdings": rows,
        "cash": {"target_weight": round(float(cash_weight), 6),
                 "drifted_weight": round(drifted_cash, 6),
                 "delta_pp": round((drifted_cash - float(cash_weight)) * 100.0, 3)},
        "note": "One rebalance period of committed price history applied to today's "
                "weights. Deltas are percentage points of weight, not returns.",
    }


# ------------------------------------------------------------------ per-beta payload
def build_one(target_beta: float, names: dict, top_n: int) -> dict:
    """One pie → one JSON payload. Everything traceable to the engine, nothing invented."""
    p = build_portfolio(capital=10_000, target_beta=target_beta, top_n=top_n,
                        make_figures=False)
    # Exact weights, not the 4dp-rounded display copy — a donut whose slices sum to 99.98%
    # is a visible defect, and the frontend has no way to correct what the bundle got wrong.
    weights = pd.Series(p["weights_exact"], dtype="float64")
    cash_weight = float(p["cash_weight_exact"])
    port = p["monthly_returns"]
    bench = p["benchmark_monthly_returns"]

    holdings = []
    for tk, w in weights.items():
        sl = p["slices"].get(tk, {})
        holdings.append({
            "ticker": tk,
            "name": names.get(tk, tk),
            "sector": sl.get("sector", "?"),
            "weight": round(float(w), 6),
            "beta": sl.get("beta"),
            "reasons": [sl.get("reason", "—")],
            # How close this holding sits to the 8% single-name cap, for the UI's cap bar.
            "cap": NAME_CAP,
            "pct_of_cap": round(float(w) / NAME_CAP, 4),
            "at_cap": bool(float(w) >= NAME_CAP - 1e-9),
        })

    sc = p["scorecard"]
    scorecard = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                 for k, v in sc.items() if not isinstance(v, (list, dict, np.ndarray))}

    return {
        "schema_version": SCHEMA_VERSION,
        "target_beta": round(float(target_beta), 2),
        "achieved_beta": p["achieved_beta"],
        "beta_capped": bool(abs(p["achieved_beta"] - target_beta) > 0.02),
        "cash_weight": round(cash_weight, 6),
        "n_holdings": int(len(weights)),
        "as_of": str(p["as_of"]),
        "benchmark": p["benchmark"],
        "holdings": holdings,
        "scorecard": scorecard,
        "equity": _series(port.index, equity_curve(port.to_numpy(dtype="float64"))),
        "drawdown": _series(port.index, _drawdown(port)),
        "beta_in_drawdown_window": beta_in_drawdown(port, bench),
        "drift": drift_after_one_period(weights, p.get("stock_monthly_returns"),
                                        cash_weight),
        "notes": p["notes"],
    }


# ------------------------------------------------------------------ validation
def validate(payload: dict) -> list[str]:
    """Schema + invariant checks. A bundle that fails these must not ship — the frontend
    trusts it completely, so this is the last place an error can be caught."""
    errs = []
    hold = payload["holdings"]
    total = sum(h["weight"] for h in hold) + payload["cash_weight"]
    if abs(total - 1.0) > 1e-4:
        errs.append(f"beta {payload['target_beta']}: weights+cash = {total:.6f}, expected 1")
    for h in hold:
        if h["weight"] > NAME_CAP + 1e-6:
            errs.append(f"beta {payload['target_beta']}: {h['ticker']} weight "
                        f"{h['weight']:.4f} exceeds the {NAME_CAP:.0%} cap")
        if not h["reasons"] or not h["reasons"][0]:
            errs.append(f"beta {payload['target_beta']}: {h['ticker']} has no reason text")
    sectors: dict[str, float] = {}
    for h in hold:
        sectors[h["sector"]] = sectors.get(h["sector"], 0.0) + h["weight"]
    for sec, w in sectors.items():
        if w > SECTOR_CAP + 1e-6:
            errs.append(f"beta {payload['target_beta']}: sector {sec} weight {w:.4f} "
                        f"exceeds the {SECTOR_CAP:.0%} cap")
    if payload["drift"].get("available"):
        for r in payload["drift"]["holdings"]:
            got = r["target_weight"] + r["delta_pp"] / 100.0
            if abs(got - r["drifted_weight"]) > 1e-6:
                errs.append(f"beta {payload['target_beta']}: {r['ticker']} drift delta "
                            f"inconsistent (target+delta != drifted)")
    if not payload["equity"]:
        errs.append(f"beta {payload['target_beta']}: empty equity series")
    return errs


def validate_as_of(label: str, as_of, today=None) -> list[str]:
    """An as-of stamp must exist and must not be in the future (#24).

    A future as-of date is the worst kind of wrong: it is the one field a user reads to decide
    how current everything else is, it looked plausible on the live site for a full release,
    and nothing crashed. So the export fails on it rather than warning.
    """
    if not as_of:
        return [f"{label}: missing as_of stamp"]
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    try:
        stamp = pd.Timestamp(as_of)
    except (ValueError, TypeError):
        return [f"{label}: as_of {as_of!r} is not a date"]
    if stamp > today:
        return [f"{label}: as_of {stamp.date()} is in the FUTURE "
                f"(today {today.date()}) — a resampled month-end label, not a data date"]
    return []


def validate_stocks(s: dict) -> list[str]:
    """stocks.json invariants — the basket page trusts this completely."""
    errs = validate_as_of("stocks.json", s.get("as_of"))
    n_dates = len(s.get("dates", []))
    if n_dates == 0:
        errs.append("stocks.json: empty date axis")
    if len(s.get("benchmark_returns", [])) != n_dates:
        errs.append("stocks.json: benchmark_returns length != dates length")
    for tk, arr in s.get("returns", {}).items():
        if len(arr) != n_dates:            # every series must align to the shared axis
            errs.append(f"stocks.json: {tk} series length {len(arr)} != {n_dates}")
        if tk not in s.get("stats", {}):
            errs.append(f"stocks.json: {tk} has a series but no stats")
    if not s.get("returns"):
        errs.append("stocks.json: no scored stocks emitted")
    return errs


def validate_explore(e: dict) -> list[str]:
    """explore.json invariants."""
    errs = validate_as_of("explore.json", e.get("as_of"))
    if not e.get("rows"):
        errs.append("explore.json: no rows")
    for r in e.get("rows", []):
        if r.get("sector") in (None, "", "?"):
            errs.append(f"explore.json: {r.get('ticker')} has no sector (caps plumbing #20)")
            break
        if r.get("cap_bucket") not in ("mid", "large"):
            errs.append(f"explore.json: {r.get('ticker')} bad cap_bucket {r.get('cap_bucket')}")
            break
    for r in e.get("rows", []):
        # A flagged row must not be offered to the basket, and a refusal must say why.
        if r.get("stat_quality") not in ("ok", "unreliable"):
            errs.append(f"explore.json: {r.get('ticker')} bad stat_quality "
                        f"{r.get('stat_quality')!r}")
            break
        if r.get("stat_quality") == "unreliable" and r.get("scored"):
            errs.append(f"explore.json: {r.get('ticker')} is flagged unreliable but still "
                        f"marked basket-eligible")
            break
        if not r.get("scored") and not r.get("scored_reason"):
            errs.append(f"explore.json: {r.get('ticker')} is not scored but carries no reason")
            break
    mv = e.get("movers", {})
    if not mv.get("winners"):
        errs.append("explore.json: no winners in movers")
    for r in mv.get("winners", []) + mv.get("losers", []):
        if r.get("stat_quality") != "ok":
            errs.append(f"explore.json: mover {r.get('ticker')} has unreliable stats")
            break
    return errs


def validate_scored_consistency(explore: dict, stocks: dict) -> list[str]:
    """Every name Explore calls basket-eligible must have a series on the basket page.

    This is the invariant that broke in #23: `scored` was set-membership in the S&P ticker
    list, while `stocks.json` additionally required 24 months of history, so Q and SNDK got a
    "basket →" button leading to a page with no data for them. Asserting it here means the
    two files cannot drift apart again without failing the export.
    """
    errs = []
    have = set(stocks.get("returns", {}))
    claimed = {r["ticker"] for r in explore.get("rows", []) if r.get("scored")}
    orphans = sorted(claimed - have)
    if orphans:
        errs.append(f"explore.json marks {len(orphans)} names basket-eligible with no series "
                    f"in stocks.json: {', '.join(orphans[:10])}")
    rec = explore.get("scored_reconciliation", {})
    if rec:
        total = (rec.get("with_explore_row", 0)
                 + rec["dropped"]["absent_from_wide_universe"])
        if total != rec.get("model_universe"):
            errs.append(f"explore.json reconciliation does not add up: "
                        f"{rec.get('with_explore_row')} with a row + "
                        f"{rec['dropped']['absent_from_wide_universe']} absent != "
                        f"{rec.get('model_universe')} in the model universe")
        if rec.get("basket_eligible") != explore.get("n_scored"):
            errs.append(f"explore.json reconciliation basket_eligible "
                        f"{rec.get('basket_eligible')} != n_scored {explore.get('n_scored')}")
    return errs


# ------------------------------------------------------------------ #23 basket (S&P scored)
def _fin(x):
    """Round a float for JSON, or None if non-finite — the frontend renders None as '—'."""
    if x is None:
        return None
    f = float(x)
    return round(f, 6) if np.isfinite(f) else None


def panel_data_date(panel: pd.DataFrame) -> str:
    """The true last DATE PRESENT IN THE DATA, e.g. '2026-06-16'.

    Not the same thing as the last label on the monthly axis. `_monthly_returns` resamples
    with `.last()` on 'ME', so a panel ending 2026-07-22 produces a final bucket *labelled*
    2026-07-31 — a date that has not happened yet. Stamping that as the as-of date tells the
    user the snapshot is more current than it is, and in the wide-universe case printed a
    future month on a live page. The panel's own max date is the only honest answer.
    """
    return pd.Timestamp(panel["date"].max()).strftime("%Y-%m-%d")


def band_flags(ann_vol, beta) -> list[str]:
    """Sanity-band violations for one name's displayed risk stats. Empty list = in band."""
    flags = []
    if beta is None or not np.isfinite(float(beta)):
        flags.append("beta_unavailable")
    elif abs(float(beta)) > BETA_ABS_MAX:
        flags.append("beta_out_of_band")
    if ann_vol is None or not np.isfinite(float(ann_vol)):
        flags.append("vol_unavailable")
    elif float(ann_vol) > ANN_VOL_MAX:
        flags.append("vol_out_of_band")
    return flags


def history_flags(col: pd.Series, axis_end) -> list[str]:
    """Sufficiency + continuity of one name's monthly return series. Empty list = usable.

    "≥24 observations" is counted as an unbroken run ending at the most recent month, not as
    a total. A 2025 spinoff (SNDK: 17 months) and a name that stopped trading in 2023 are
    both unusable, and a total-count test would pass the second one.
    """
    valid = col.dropna()
    if valid.empty:
        return ["no_history"]
    flags = []
    first, last = col.first_valid_index(), col.last_valid_index()
    if int(col.loc[first:last].isna().sum()) > 0:
        flags.append("discontinuous")
    if last != axis_end:
        flags.append("stale_history")
    n_trailing = 0
    for v in reversed(col.to_numpy(dtype="float64").tolist()):
        if not np.isfinite(v):
            break
        n_trailing += 1
    if n_trailing < BASKET_MIN_HISTORY:
        flags.append("insufficient_history")
    return flags


def reason_text(flags: list[str]) -> str | None:
    """Human-readable 'why not' for a list of flags, or None when there is nothing to say."""
    if not flags:
        return None
    return "; ".join(FLAG_REASONS.get(f, f) for f in flags)


def _stock_stats(r: pd.Series, bench: pd.Series) -> dict:
    """Per-stock summary, analyser conventions (ddof=0, PPY=12). r, bench aligned monthly."""
    r = r.dropna()
    eq = equity_curve(r.to_numpy(dtype="float64"))
    mdd, _ = max_drawdown(eq)
    b = _beta(r, bench.reindex(r.index)) if len(r) >= 2 else float("nan")
    return {
        "n": int(len(r)),
        "total_return": _fin(total_return(r)),
        "ann_vol": _fin(volatility(r)),
        "sharpe": _fin(sharpe(r)),
        "sortino": _fin(sortino(r)),
        "max_drawdown": _fin(mdd),
        "beta": _fin(b),
        "last_return": _fin(float(r.iloc[-1])) if len(r) else None,
    }


def build_stocks(scored: set[str], names: dict) -> dict:
    """Per-stock monthly return series + summary stats for the S&P-500 SCORED names — the
    raw material the basket page does all its math on, client-side. Shared date axis +
    per-ticker return arrays (nulls where a name lacks that month) keeps it compact. The
    benchmark is the equal-weight S&P proxy — identical to the pie's, so basket-vs-pie is
    apples to apples.

    #24: eligibility is gated here and NOWHERE ELSE, and every refusal is recorded with its
    reason in `excluded`. The Explore tab consumes this same decision, so the table can never
    again offer a "basket →" button for a name the basket page has no series for.
    """
    panel = pd.read_parquet(SP500_PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    rets = _monthly_returns(panel).dropna(how="all").tail(MAX_LOOKBACK)  # ≤72 month-ends
    bench = rets.mean(axis=1, skipna=True)                               # EW S&P proxy
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in rets.index]
    axis_end = rets.index[-1] if len(rets.index) else None

    def arr(s: pd.Series):
        return [None if not np.isfinite(float(v)) else round(float(v), 6) for v in s]

    returns, stats, excluded = {}, {}, {}
    for tk in sorted(scored):
        if tk not in rets.columns:
            excluded[tk] = {"flags": ["no_history"],
                            "reason": FLAG_REASONS["no_history"]}
            continue
        col = rets[tk]
        flags = history_flags(col, axis_end)
        st = _stock_stats(col, bench)
        flags = flags + band_flags(st["ann_vol"], st["beta"])
        if flags:
            excluded[tk] = {"flags": flags, "reason": reason_text(flags)}
            continue
        returns[tk] = arr(col.reindex(rets.index))
        st["name"] = names.get(tk, tk)
        stats[tk] = st

    return {
        "schema_version": SCHEMA_VERSION,
        # The true last data date, not the month-end bucket label (#24).
        "as_of": panel_data_date(panel),
        "axis_end_label": dates[-1] if dates else None,
        "axis_last_month_partial": bool(
            axis_end is not None and pd.Timestamp(panel["date"].max()) < axis_end),
        "benchmark_label": "equal-weight S&P-500 proxy",
        "periods_per_year": 12,
        "dates": dates,
        "benchmark_returns": arr(bench),
        "returns": returns,       # {ticker: [r0..rN] aligned to dates, null for missing}
        "stats": stats,           # {ticker: {ann_vol, beta, sharpe, sortino, ...}}
        "excluded": excluded,     # {ticker: {flags, reason}} — why a scored name is absent
        "eligibility": {
            "min_history_months": BASKET_MIN_HISTORY,
            "beta_abs_max": BETA_ABS_MAX,
            "ann_vol_max": ANN_VOL_MAX,
            "text": f"A name is basket-eligible only with ≥{BASKET_MIN_HISTORY} unbroken "
                    f"recent monthly returns, |beta| ≤ {BETA_ABS_MAX:.0f} and annualised "
                    f"volatility ≤ {ANN_VOL_MAX:.0%}.",
        },
        "note": "Equal-weight basket math is done client-side from these series. Historical "
                "characterisation only — never a forecast. The final month may be partial: "
                "see axis_last_month_partial.",
    }


# ------------------------------------------------------------------ #23 explore (wide universe)
def build_explore(eligible: set[str], model_universe: set[str],
                  exclusions: dict) -> dict | None:
    """One row per wide-universe (1,200) name for the Explore tab: identity, sector (#20),
    cap bucket, last-period return, ann vol, realised beta — all as of the bundle date, never
    live. Returns None if the wide-universe build inputs are absent (fresh checkout without
    `make universe`/`make sectors`); the frontend then simply hides the tab.

    `eligible` is the basket-eligible set decided by `build_stocks` — this function does not
    get its own opinion about eligibility, which is what keeps the two files consistent.
    Stats displayed here are measured on the WIDE panel against the wide-universe benchmark,
    so they are sanity-banded here too: a row can be flagged `unreliable` for display while
    the same name remains basket-eligible on the S&P panel, and both statements are true.
    """
    if not (MIDLARGE_PANEL.exists() and MIDLARGE_UNIVERSE.exists() and MIDLARGE_SECTORS.exists()):
        logger.warning("wide-universe inputs absent — skipping explore.json "
                       "(run `make universe` + `make sectors` to enable Explore)")
        return None

    uni = pd.read_csv(MIDLARGE_UNIVERSE)
    secmap = pd.read_csv(MIDLARGE_SECTORS).set_index("ticker")["sector"].astype(str)
    caps = uni.set_index("ticker")["market_cap"] if "market_cap" in uni else pd.Series(dtype=float)
    company = uni.set_index("ticker")["name"] if "name" in uni else pd.Series(dtype=object)

    panel = pd.read_parquet(MIDLARGE_PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    rets = _monthly_returns(panel).dropna(how="all").tail(MAX_LOOKBACK)
    bench = rets.mean(axis=1, skipna=True)
    axis_end = rets.index[-1] if len(rets.index) else None

    rows = []
    for tk in uni["ticker"]:
        if tk not in rets.columns:
            continue
        col = rets[tk]
        r = col.dropna()
        if r.empty:
            continue
        cap = float(caps.get(tk, float("nan")))
        ann_vol = _fin(volatility(r))
        bta = _fin(_beta(r, bench.reindex(r.index)) if len(r) >= 2 else float("nan"))
        # Display band (measured on THIS panel) + basket eligibility (decided on the S&P
        # panel). A row is only offered to the basket if it passes both.
        bflags = band_flags(ann_vol, bta)
        hflags = history_flags(col, axis_end)
        is_eligible = bool(tk in eligible) and not bflags
        # Every reason that applies, not just the first one found. A name can be both outside
        # the model's universe AND carrying a junk beta, and "not scored by the model" alone
        # would leave the user thinking the 364 in the beta column is a real measurement.
        why_parts = []
        if tk not in model_universe:
            why_parts.append(FLAG_REASONS["not_in_model_universe"])
        elif tk not in eligible:
            why_parts.append(exclusions.get(tk, {}).get("reason")
                             or reason_text(hflags + bflags) or "")
        if bflags:
            why_parts.append(reason_text(bflags))
        # The S&P-panel refusal may already name the same band violation the wide panel sees;
        # say each clause once, in the order it was added.
        seen, clauses = set(), []
        for part in why_parts:
            for clause in (part or "").split("; "):
                if clause and clause not in seen:
                    seen.add(clause)
                    clauses.append(clause)
        why = "; ".join(clauses) or None
        rows.append({
            "ticker": tk,
            "name": str(company.get(tk, tk)),
            "sector": str(secmap.get(tk, "?")),
            "cap_bucket": ("large" if np.isfinite(cap) and cap >= CAP_LARGE_USD else "mid"),
            "market_cap": None if not np.isfinite(cap) else round(cap, 0),
            "last_return": _fin(float(r.iloc[-1])),
            "ann_vol": ann_vol,
            "beta": bta,
            # Numbers above are NEVER winsorised. This says "do not trust them", and the UI
            # keeps flagged rows out of the default beta/vol orderings.
            "stat_quality": "unreliable" if bflags else "ok",
            "stat_flags": bflags,
            "stat_note": reason_text(bflags),
            "n_months": int(len(r)),
            "scored": is_eligible,
            "scored_reason": why,
        })

    # Movers rank on last-month return, which the band does not police — but a row whose
    # stats are junk has no business fronting the page, so flagged rows are excluded from
    # the cards and the count says so.
    clean = [r for r in rows if r["last_return"] is not None and r["stat_quality"] == "ok"]
    ranked = sorted(clean, key=lambda r: r["last_return"], reverse=True)
    movers = {"winners": ranked[:MOVERS_N],
              "losers": list(reversed(ranked[-MOVERS_N:])) if len(ranked) >= MOVERS_N else [],
              "excluded_unreliable": sum(1 for r in rows if r["stat_quality"] != "ok"),
              "note": "Ranked on last-month return across rows whose risk stats pass the "
                      "sanity band."}

    row_tickers = {r["ticker"] for r in rows}
    return {
        "schema_version": SCHEMA_VERSION,
        # True last data date — never the resampled month-end label, which is in the future
        # for any panel that ends mid-month (#24).
        "as_of": panel_data_date(panel),
        "axis_end_label": (pd.Timestamp(axis_end).strftime("%Y-%m-%d") if axis_end is not None
                           else None),
        "axis_last_month_partial": bool(
            axis_end is not None and pd.Timestamp(panel["date"].max()) < axis_end),
        "benchmark_label": "equal-weight wide-universe proxy (beta reference)",
        "cap_threshold_usd": CAP_LARGE_USD,
        "n_names": len(rows),
        "n_scored": sum(1 for r in rows if r["scored"]),
        "n_unreliable": sum(1 for r in rows if r["stat_quality"] != "ok"),
        "sanity_band": {"beta_abs_max": BETA_ABS_MAX, "ann_vol_max": ANN_VOL_MAX,
                        "text": f"Rows with |beta| > {BETA_ABS_MAX:.0f} or annualised "
                                f"volatility > {ANN_VOL_MAX:.0%} are flagged, kept out of "
                                f"the default risk orderings, and not basket-eligible. The "
                                f"displayed figures are the measured ones, unaltered."},
        "scored_reconciliation": _reconcile(model_universe, row_tickers, eligible,
                                            exclusions, rows),
        "rows": rows,
        "movers": movers,
        "caveat": "Snapshot as of the bundle data date — NOT live/today. Small caps (<$2B) "
                  "are excluded by universe methodology.",
    }


def _reconcile(model_universe: set[str], row_tickers: set[str], eligible: set[str],
               exclusions: dict, rows: list[dict]) -> dict:
    """The audited funnel from "the model ranks 503 names" to the number Explore shows.

    #24 asked where ~50 names went between the S&P set and `n_scored`. Almost all of them
    never reach Explore at all: they are missing from the wide-universe market-cap screen
    because `universe.shares_outstanding` reads the SEC XBRL *frames* API, which returns only
    undimensioned facts — and multi-class filers tag share counts per share class, so GOOGL,
    META, BRK-B, V, MA, F and ~44 others have no share count in the frame and were dropped
    before the price screen ever ran. See `universe.py` (fixed there for future builds; the
    shipped panel predates the fix and still omits them).
    """
    in_wide = model_universe & row_tickers
    missing_wide = sorted(model_universe - row_tickers)
    band_dropped = sorted(r["ticker"] for r in rows
                          if r["ticker"] in eligible and r["stat_quality"] != "ok")
    hist_dropped = sorted(tk for tk in in_wide
                          if tk not in eligible and tk in exclusions)
    return {
        "model_universe": len(model_universe),
        "with_explore_row": len(in_wide),
        "basket_eligible": len(in_wide & eligible) - len(band_dropped),
        "eligible_without_explore_row": len(eligible - row_tickers),
        "dropped": {
            "absent_from_wide_universe": len(missing_wide),
            "failed_history_or_stats_on_sp_panel": len(hist_dropped),
            "failed_wide_panel_sanity_band": len(band_dropped),
        },
        "absent_from_wide_universe_tickers": missing_wide,
        "failed_history_tickers": hist_dropped,
        "failed_band_tickers": band_dropped,
        "why_absent_from_wide_universe":
            "The wide universe screens SEC registrants for a share count via the XBRL "
            "`frames` API, which only returns facts carrying no dimensions. Multi-class "
            "filers tag EntityCommonStockSharesOutstanding per share class, so their share "
            "count is invisible to that endpoint and they were dropped before the market-cap "
            "filter ran. This is a universe-construction bug, not a data-quality property of "
            "the names — it is fixed in `universe.py` for future builds, and the shipped "
            "panel still predates the fix.",
        "note": f"{len(eligible - row_tickers)} basket-eligible names have no Explore row at "
                f"all, for the same reason — the Basket tab reaches them, the table does not.",
    }


# ------------------------------------------------------------------ #23 regime beta (from #21)
def build_regime() -> dict | None:
    """The MEASURED calm vs stressed-core beta per preset, straight from the #21 regime
    backtest — so the receipts caveat quotes a number, not a hand-wave. None if the lab
    module or its data is unavailable (the frontend then shows only the generic caveat)."""
    try:
        from lab.regime_backtest import run as regime_run
    except Exception as e:  # noqa: BLE001
        logger.warning("regime beta unavailable (%s) — receipts show generic caveat only", e)
        return None
    try:
        res = regime_run(make_report=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("regime backtest failed (%s) — skipping regime beta", e)
        return None
    stats, core = res["stats"], res["core_stats"]
    label_for = {0.50: "Pie β0.50", 0.75: "Pie β0.75", 1.00: "Pie β1.00"}
    out = {}
    for b, label in label_for.items():
        if label in stats:
            out[f"{b:.2f}"] = {
                "calm": _fin(stats[label]["calm"]["beta"]),
                "stressed_core": _fin(core[label]["beta"]),
                "stressed_full": _fin(stats[label]["stressed"]["beta"]),
                "n_core": core[label]["n"],
                "core_months": res.get("drawdown_driven_months", []),
            }
    return out or None


# ------------------------------------------------------------------ main
def export(out_dir: Path = OUT_DIR, top_n: int = 20, betas=None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "beta").mkdir(parents=True, exist_ok=True)

    names = {}
    if TICKERS_CSV.exists():
        meta = pd.read_csv(TICKERS_CSV)
        if {"ticker", "name"}.issubset(meta.columns):
            names = dict(zip(meta["ticker"], meta["name"]))

    # Ask the engine what it can actually reach, rather than trusting #15's example number.
    probe = build_portfolio(capital=10_000, target_beta=ABSURD_BETA, top_n=top_n,
                            make_figures=False)
    max_beta = float(probe["achieved_beta"])
    logger.info("engine max achievable long-only beta = %.3f", max_beta)

    grid = betas if betas is not None else beta_grid()
    errors, written = [], []
    for i, b in enumerate(grid):
        payload = build_one(b, names, top_n)
        errors.extend(validate(payload))
        (out_dir / "beta" / f"{beta_key(b)}.json").write_text(
            json.dumps(payload, separators=(",", ":")))
        written.append(beta_key(b))
        logger.info("[%2d/%2d] beta %.2f → achieved %.2f · cash %.1f%% · %d holdings",
                    i + 1, len(grid), b, payload["achieved_beta"],
                    payload["cash_weight"] * 100, payload["n_holdings"])

    # Benchmark series once, globally — it is identical for every pie, so per-beta copies
    # would be pure duplication in every payload the browser downloads.
    bp = build_portfolio(capital=10_000, target_beta=1.0, top_n=top_n, make_figures=False)
    bench = bp["benchmark_monthly_returns"]
    benchmark = {
        "label": bp["benchmark"],
        "equity": _series(bench.index, equity_curve(bench.to_numpy(dtype="float64"))),
        "drawdown": _series(bench.index, _drawdown(bench)),
    }

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_from": "portfolio.beta_engine.build_portfolio (S&P 500 frozen model)",
        "as_of": str(bp["as_of"]),
        "betas": grid,
        "beta_keys": {f"{b:.2f}": beta_key(b) for b in grid},
        "beta_step": BETA_STEP,
        "max_achievable_beta": round(max_beta, 3),
        "presets": [{"label": lbl, "beta": b, "key": beta_key(b)} for lbl, b in PRESETS],
        "default_capital": 10000,
        "currency": "GBP",
        "benchmark": benchmark,
        "guardrails": {
            "name_cap": NAME_CAP,
            "sector_cap": SECTOR_CAP,
            "sector_max_names": SECTOR_MAX_NAMES,
            "text": "max 8% per stock · max 30% per sector · ≤5 stocks per sector · "
                    "risk-balanced sizing",
        },
        "how_it_works": HOW_IT_WORKS,
        "caveats": CAVEATS,
        "regime_beta": build_regime(),   # #21 measured calm vs stressed-core beta per preset
        "disclaimer": "Educational simulation. No real money. Not investment advice.",
    }
    errors.extend(validate_as_of("index.json", index["as_of"]))
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    # #23 — the S&P-500 scored universe (basket-eligible). The frozen model ranks these;
    # it can't speak for names it never trained on, so only these get a per-stock series.
    model_universe = set(names) if names else set()
    if TICKERS_CSV.exists():
        model_universe = set(pd.read_csv(TICKERS_CSV)["ticker"])

    stocks = build_stocks(model_universe, names)
    errors.extend(validate_stocks(stocks))
    (out_dir / "stocks.json").write_text(json.dumps(stocks, separators=(",", ":")))
    eligible = set(stocks["returns"])
    logger.info("basket-eligible: %d of %d model-universe names (%d refused: %s)",
                len(eligible), len(model_universe), len(stocks["excluded"]),
                ", ".join(sorted(stocks["excluded"])[:8]) or "none")

    explore = build_explore(eligible, model_universe, stocks["excluded"])
    if explore is not None:
        errors.extend(validate_explore(explore))
        errors.extend(validate_scored_consistency(explore, stocks))
        (out_dir / "explore.json").write_text(json.dumps(explore, separators=(",", ":")))

    size = sum(f.stat().st_size for f in out_dir.rglob("*.json"))
    return {"out_dir": str(out_dir), "n_betas": len(written), "errors": errors,
            "max_achievable_beta": max_beta, "bytes": size,
            "index_bytes": (out_dir / "index.json").stat().st_size,
            "stocks_bytes": (out_dir / "stocks.json").stat().st_size,
            "explore_bytes": ((out_dir / "explore.json").stat().st_size
                              if (out_dir / "explore.json").exists() else 0),
            "n_scored": len(stocks["returns"]),
            "n_refused": len(stocks["excluded"]),
            "reconciliation": (explore or {}).get("scored_reconciliation"),
            "n_unreliable": (explore or {}).get("n_unreliable", 0),
            "as_of": {"index": index["as_of"], "stocks": stocks["as_of"],
                      "explore": (explore or {}).get("as_of")},
            "n_explore": explore["n_names"] if explore else 0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the #19 static web bundle")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--betas", default=None,
                    help="comma-separated betas (smoke test), e.g. 0.5,1.0")
    args = ap.parse_args()

    betas = ([round(float(x), 2) for x in args.betas.split(",")] if args.betas else None)
    res = export(Path(args.out), top_n=args.top_n, betas=betas)

    print(f"\nwrote {res['n_betas']} beta files → {res['out_dir']}")
    print(f"as-of stamps: index {res['as_of']['index']} · stocks {res['as_of']['stocks']} "
          f"· explore {res['as_of']['explore']}  (all must be <= today)")
    print(f"stocks.json: {res['n_scored']} basket-eligible S&P names, {res['n_refused']} "
          f"refused ({res['stocks_bytes']/1024:.1f} KB)")
    print(f"explore.json: {res['n_explore']} wide-universe rows, {res['n_unreliable']} "
          f"flagged unreliable ({res['explore_bytes']/1024:.1f} KB)")
    rec = res.get("reconciliation")
    if rec:
        print(f"  scored funnel: {rec['model_universe']} model universe → "
              f"{rec['with_explore_row']} with an Explore row → "
              f"{rec['basket_eligible']} basket-eligible")
        for k, v in rec["dropped"].items():
            print(f"    - {v:3d} dropped: {k}")
    print(f"max achievable long-only beta (engine-computed): {res['max_achievable_beta']:.3f}")
    print(f"bundle size: {res['bytes']/1024:.1f} KB "
          f"(index {res['index_bytes']/1024:.1f} KB) — budget ~5 MB")
    if res["errors"]:
        print(f"\n❌ {len(res['errors'])} VALIDATION ERRORS — bundle must not ship:")
        for e in res["errors"][:20]:
            print("  -", e)
        return 1
    print("\n✅ all bundle invariants hold (weights sum to 1, caps respected, "
          "drift deltas consistent, reasons present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
