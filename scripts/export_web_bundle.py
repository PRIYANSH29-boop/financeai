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

from analytics.metrics import beta as _beta, equity_curve  # noqa: E402
from portfolio.beta_engine import (  # noqa: E402
    build_portfolio, NAME_CAP, SECTOR_CAP, SECTOR_MAX_NAMES,
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
        "disclaimer": "Educational simulation. No real money. Not investment advice.",
    }
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    size = sum(f.stat().st_size for f in out_dir.rglob("*.json"))
    return {"out_dir": str(out_dir), "n_betas": len(written), "errors": errors,
            "max_achievable_beta": max_beta, "bytes": size,
            "index_bytes": (out_dir / "index.json").stat().st_size}


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
    print(f"max achievable long-only beta (engine-computed): {res['max_achievable_beta']:.3f}")
    print(f"bundle size: {res['bytes']/1024:.1f} KB "
          f"(index {res['index_bytes']/1024:.1f} KB)")
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
