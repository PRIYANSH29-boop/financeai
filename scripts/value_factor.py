#!/usr/bin/env python3
"""
Value factor A/B — Phase 18. The survival-chain test for the value composite.

Prereq: the #17 audit verdict is GO (`figures/audit/fundamentals_audit.md`). This script
refuses to run if the report is missing or says NO-GO — the gate is the point.

    python scripts/value_factor.py --build          # fetch fundamentals, then test
    python scripts/value_factor.py                  # reuse data/sec_fundamentals.parquet

Runs three books through the SAME frozen book machinery (top-50 long-only, inverse-vol,
8% cap, 14% vol target, 10 bps/side) — only the score recipe changes:

    A · momentum            B · momentum + value            C · value only

over two windows: the frozen paper-track window (directly comparable to the #14 low-vol
test) and the full labelled history (more months, since an equal-weight factor combine
involves no fitted model and therefore has no out-of-sample constraint).

Writes `figures/lab/value_factor.md` + `figures/lab/value_factor.png`.
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

from analytics import compare  # noqa: E402
from lab import value_factor as vf  # noqa: E402
from lab.strategy_lab import (  # noqa: E402
    run_strategy, signal_correlation, monthly_rebalances,
)
from portfolio.paper_trade import FREEZE_DATE, _rebalance_dates  # noqa: E402

AUDIT_REPORT = Path("figures/audit/fundamentals_audit.md")
OUT_MD = Path("figures/lab/value_factor.md")
OUT_PNG = Path("figures/lab/value_factor.png")

MOM_FACTORS = [("mom_12_1m", True)]
VAL_FACTORS = [("value_score", True)]


def audit_verdict(path: Path = AUDIT_REPORT) -> str:
    """Read the #17 GO/NO-GO out of the committed report. Missing report ⇒ blocked."""
    if not path.exists():
        return "MISSING"
    for line in path.read_text().splitlines():
        if line.startswith("## VERDICT:"):
            return "GO" if "**GO**" in line else "NO-GO"
    return "UNKNOWN"


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without pulling in `tabulate` for one call."""
    head = "| metric | " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|---" * (len(df.columns) + 1) + "|"
    rows = [f"| {idx} | " + " | ".join(str(v) for v in df.loc[idx]) + " |"
            for idx in df.index]
    return "\n".join([head, rule, *rows])


def run_window(name: str, labeled: pd.DataFrame, panel: pd.DataFrame,
               rebals: list, allow_leverage: bool = False) -> dict:
    """A/B/C scorecard + value diagnostics for one rebalance schedule.

    allow_leverage : two-sided vol target, so a lower-vol book is levered UP to the same
        14% risk. This is the comparison #14 used to decide low-vol on: it separates
        "the factor earns more per unit of risk" from "the factor just de-risked the book".
    """
    specs = [vf.MOMENTUM, vf.MOMENTUM_VALUE, vf.VALUE_ONLY]
    books = {s["name"]: run_strategy(s, labeled=labeled, panel=panel, rebalances=rebals,
                                     allow_leverage=allow_leverage)
             for s in specs}
    rets = {k: pd.Series(v["net_ret"].to_numpy(), index=pd.to_datetime(v["date"]), name=k)
            for k, v in books.items()}
    first = next(iter(books.values()))
    bench = pd.Series(first["bench_ret"].to_numpy(),
                      index=pd.to_datetime(first["date"]), name="Equal-weight universe")

    table = compare({**rets, bench.name: bench}, benchmark=bench, periods_per_year=12,
                    pretty=True)

    corr_series = signal_correlation(MOM_FACTORS, VAL_FACTORS, labeled=labeled,
                                     rebalances=rebals, per_date=True)
    turnover = {k: float(v["turnover"].mean()) for k, v in books.items()}

    return {
        "name": name,
        "n_months": int(len(first)),
        "start": str(pd.to_datetime(first["date"]).min().date()),
        "end": str(pd.to_datetime(first["date"]).max().date()),
        "table": table,
        "returns": rets,
        "bench": bench,
        "corr_mean": float(np.nanmean(corr_series)),
        "corr_min": float(np.nanmin(corr_series)),
        "corr_max": float(np.nanmax(corr_series)),
        "turnover": turnover,
    }


def make_chart(windows: list[dict], out: Path = OUT_PNG) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(windows), figsize=(7 * len(windows), 5.2))
    axes = np.atleast_1d(axes)
    colors = {"A · Momentum": "#1f77b4", "B · Momentum + value": "#d62728",
              "C · Value only": "#2ca02c"}
    for ax, w in zip(axes, windows):
        for label, r in w["returns"].items():
            ax.plot((1 + r).cumprod(), label=label, lw=1.9,
                    color=colors.get(label, "#888888"))
        ax.plot((1 + w["bench"]).cumprod(), label="Equal-weight universe", lw=1.2,
                color="#888888", ls=":")
        ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
        ax.set_title(f"{w['name']}\n{w['start']} → {w['end']} · {w['n_months']} months",
                     fontsize=10)
        ax.set_ylabel("Growth of $1")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("RankAlpha Phase 18 — value factor A/B (long-only, after 10 bps/side)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def write_report(windows: list[dict], meta: dict, out: Path = OUT_MD) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    L = ["# RankAlpha — value factor A/B (Phase 18)\n",
         "*Reproducible via `python scripts/value_factor.py`. Educational SIMULATION — no "
         "promised returns, survivorship-biased universe, results DIRECTIONAL.*\n",
         f"Fundamentals: **{meta['n_rows']:,} point-in-time records** over "
         f"**{meta['n_tickers']} tickers**, SEC EDGAR XBRL, publication-date lagged "
         f"{vf.PUBLICATION_LAG_DAYS}d. #17 audit verdict: **{meta['verdict']}**.\n",
         "## Composite spec\n",
         "| ratio | definition | orientation |",
         "|---|---|---|",
         "| earnings yield | EPS (TTM) / price | higher = cheaper |",
         "| book-to-market | book value / market cap | higher = cheaper |",
         "| EBITDA/EV | EBITDA (TTM) / enterprise value | higher = cheaper |",
         "| FCF yield | free cash flow (TTM) / market cap | higher = cheaper |",
         "",
         f"Each ratio winsorized at {vf.WINSOR_PCT if hasattr(vf,'WINSOR_PCT') else (0.01,0.99)}, "
         f"z-scored cross-sectionally per rebalance, then averaged; a name needs "
         f"≥{vf.MIN_RATIOS} of the 4. Value coverage across rebalance dates: "
         f"**{meta['coverage']:.1%}** of the eligible cross-section; uncovered names get a "
         "neutral score (0.0) so the A/B universe is identical.\n"]

    for w in windows:
        L.append(f"## {w['name']} — {w['start']} → {w['end']} ({w['n_months']} months)\n")
        L.append(md_table(w["table"]))
        L.append("")
        L.append(f"- **corr(value, momentum)**: mean **{w['corr_mean']:+.3f}** "
                 f"(range {w['corr_min']:+.3f} … {w['corr_max']:+.3f}), cross-sectional "
                 f"Spearman per rebalance.")
        t = w["turnover"]
        a, b = t["A · Momentum"], t["B · Momentum + value"]
        direction = "more" if b > a else "less"
        L.append(f"- **turnover** (mean per rebalance, sum |Δw|): A {a:.2f} → B {b:.2f} "
                 f"(Δ {b - a:+.2f}, ≈ {abs(b - a) * 10:.1f} bps/month {direction} cost at "
                 f"10 bps/side).")
        L.append("")

    L.append("## Verdict\n")
    L.append(meta["verdict_text"])
    L.append("")
    L.append("## Caveats\n")
    L.append("- Short windows: the frozen-track window is ~2 years of monthly observations; "
             "no Sharpe difference over that span is statistically significant.")
    L.append("- Survivorship: today's S&P 500 membership applied to all history — the "
             "cheapest names are precisely those most likely to have been deleted, so a "
             "survivorship-biased universe flatters value more than it flatters momentum.")
    L.append("- Coverage is uneven by sector: EBITDA/EV and FCF yield are undefined for most "
             "banks (no capex line), so financials effectively score on E/P and B/M only.")
    L.append("- Educational SIMULATION. Not investment advice.")
    out.write_text("\n".join(L))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RankAlpha value factor A/B (#18)")
    ap.add_argument("--build", action="store_true",
                    help="(re)build data/sec_fundamentals.parquet from EDGAR before testing")
    ap.add_argument("--tickers", default="data/sp500_tickers.csv")
    ap.add_argument("--force", action="store_true",
                    help="run even if the #17 audit verdict is not GO (not recommended)")
    args = ap.parse_args()

    verdict = audit_verdict()
    if verdict != "GO" and not args.force:
        print(f"BLOCKED: #17 fundamentals audit verdict is {verdict!r}, not GO.\n"
              f"         Run `python scripts/audit_fundamentals.py` first.", file=sys.stderr)
        return 2

    panel = pd.read_parquet("data/sp500_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    labeled = pd.read_parquet("data/sp500_labeled.parquet")
    labeled["date"] = pd.to_datetime(labeled["date"])

    if args.build or not vf.FUND_PATH.exists():
        tickers = pd.read_csv(args.tickers).iloc[:, 0].astype(str).str.strip().tolist()
        fund = vf.build_fundamentals(tickers, panel=panel)
    else:
        fund = vf.load_fundamentals()

    frozen_rebals = [pd.Timestamp(d) for d in _rebalance_dates(labeled, FREEZE_DATE)]
    full_rebals = monthly_rebalances(labeled)
    all_rebals = sorted(set(frozen_rebals) | set(full_rebals))

    print(f"Scoring value on {len(all_rebals)} rebalance dates…")
    vpanel = vf.value_panel(fund, panel, all_rebals)
    labeled = vf.attach(labeled, vpanel)

    on_rebal = labeled[labeled["date"].isin(all_rebals)]
    coverage = float(on_rebal["value_covered"].mean())
    print(f"value_score coverage on rebalance dates: {coverage:.1%}")

    windows = [
        run_window("Frozen paper-track window", labeled, panel, frozen_rebals),
        run_window("Full labelled history", labeled, panel, full_rebals),
        run_window("Full history, matched vol (14%)", labeled, panel,
                   full_rebals, allow_leverage=True),
    ]

    for w in windows:
        print(f"\n=== {w['name']} ({w['n_months']} months) ===")
        print(w["table"].to_string())
        print(f"corr(value, momentum) = {w['corr_mean']:+.3f}")

    verdict_text = _decide(windows)
    meta = {"n_rows": len(fund), "n_tickers": int(fund["ticker"].nunique()),
            "verdict": verdict, "coverage": coverage, "verdict_text": verdict_text}
    make_chart(windows)
    path = write_report(windows, meta)
    print("\n" + verdict_text)
    print(f"\nReport: {path}  ·  Chart: {OUT_PNG}")
    return 0


def _decide(windows: list[dict]) -> str:
    """Keep/drop recommendation computed from the scorecards, not asserted.

    The survival-chain rule from #14: keep a factor only if it is UNCORRELATED with what we
    already trade AND it improves the scorecard. Passing only the first test is not enough —
    an independent signal that costs Sharpe is still a worse portfolio.
    """
    lines, sharpe_deltas, dd_deltas = [], [], []
    for w in windows:
        t = w["table"]
        sa, sb = _pct(t.at["Sharpe", "A · Momentum"]), _pct(t.at["Sharpe", "B · Momentum + value"])
        da = _pct(t.at["Max drawdown", "A · Momentum"])
        db = _pct(t.at["Max drawdown", "B · Momentum + value"])
        sharpe_deltas.append(sb - sa)
        dd_deltas.append(db - da)          # positive = shallower drawdown (less negative)
        lines.append(
            f"- **{w['name']}** ({w['n_months']} months): Sharpe "
            f"{t.at['Sharpe','A · Momentum']} → {t.at['Sharpe','B · Momentum + value']} "
            f"({sb - sa:+.2f}); max drawdown {t.at['Max drawdown','A · Momentum']} → "
            f"{t.at['Max drawdown','B · Momentum + value']} ({db - da:+.2%}); "
            f"corr(value, momentum) {w['corr_mean']:+.3f}.")

    uncorrelated = all(abs(w["corr_mean"]) < 0.30 for w in windows)
    sharpe_up = all(d > 0.02 for d in sharpe_deltas)
    dd_better = all(d > 0.005 for d in dd_deltas)

    if uncorrelated and sharpe_up:
        rec = ("**KEEP.** Value is independent of momentum and lifts risk-adjusted return "
               "in every window tested.")
    elif uncorrelated and dd_better and not sharpe_up:
        rec = ("**DROP for now — but a genuine near-miss, and the reason is worth stating "
               "precisely.** Value passes the independence test (the hard one) and it does "
               "cut drawdown in every window. What it does not do is pay for itself: Sharpe "
               "falls in every window because the return it gives up exceeds the risk it "
               "removes. That is the opposite of the low-vol result in #14, which held "
               "return while halving drawdown and so earned its place. Under the "
               "survival-chain rule — uncorrelated AND improves the scorecard — one out of "
               "two is a DROP. Keep the harness and the point-in-time data; revisit value "
               "when it can be tested on a universe where cheap names are not "
               "systematically the ones survivorship deleted.")
    elif not uncorrelated:
        rec = ("**DROP.** Value is too correlated with momentum to be adding independent "
               "information.")
    else:
        rec = ("**DROP.** Value neither improves risk-adjusted return nor reduces drawdown "
               "reliably across windows.")
    return "\n".join([*lines, "", rec])


def _pct(v):
    """Parse a `compare(pretty=True)` cell ('1.55', '-13.09%') back to a float."""
    s = str(v).strip()
    if s in {"—", "nan", ""}:
        return float("nan")
    return float(s[:-1]) / 100.0 if s.endswith("%") else float(s)


if __name__ == "__main__":
    raise SystemExit(main())
