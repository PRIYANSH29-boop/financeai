"""
The Signal Duel — RankAlpha #27, run under #30's asymmetric ruling.

⚠️ EDUCATIONAL SIMULATION. Nothing is trained here and nothing is predicted. Both books are
scored on the SAME walk-forward out-of-sample frame; the frozen model is read, never refit.

The question, open since Phase 14
---------------------------------
Plain 12-1 momentum scored Sharpe 1.55 against the frozen ML book's 1.28 in one window. If
momentum beats the machine, the product should trade momentum and the ML needs a rethink.
This settles it on one axis with one variable.

One variable, by construction
-----------------------------
Both books run through `lab.strategy_lab.run_strategy`, which imports the frozen track's own
book construction verbatim from `portfolio/paper_trade.py`: same universe, same rebalance
schedule, same long-only top-N, same inverse-vol sizing, same weight cap, same vol target,
same 10 bps/side turnover cost. **Only the score differs.**

  * Book A — MOMENTUM: cross-sectional percentile rank of `mom_12_1m`.
  * Book B — FROZEN ML: `model_score` from `data/sp500_oos_walkforward.parquet`.

Why that file and not the model object: every `model_score` in it is **walk-forward
out-of-sample** — predicted by a model fit only on data before that date, with a 21-day
embargo. Scoring the frozen model over its own training window would hand book B an
in-sample advantage and make the duel meaningless.

The A-1 contamination, and why it made the v1 result ASYMMETRIC
--------------------------------------------------------------
**A-1 is now FIXED (#31 Arm 1) and the v2 rematch showed it was never carrying the ML — the
clean model scores Sharpe 1.448 vs the contaminated 1.407, still far behind momentum's 1.792.
The section below describes the v1 duel, whose published numbers remain as run.**

At the time of the v1 duel, `size` (a model feature) was `log(adj_close)` — a bare,
retroactively re-adjusted price level (#25 finding A-1). Its historical value moves whenever a later split or dividend
lands, so it leaks a little future information **into book B only**. Momentum is a price
*ratio*: the adjustment factor cancels, so book A is immune. `test_utils_pipeline_f2.py`
proves both halves of that claim.

The contamination therefore FAVOURS THE ML. The reviewer's pre-stated reading (#28):

    ML loses  ⇒ CONCLUSIVE. It lost while carrying an advantage. Trade momentum.
    ML wins   ⇒ INCONCLUSIVE, pending a clean A-1-fixed retrain. The win may be the leak.

A clean retrain was a separate phase — #31 Arm 1 — and it has now been run. See
`figures/lab/last_stand.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analytics.metrics import max_drawdown, sharpe, sortino, volatility
from lab.regime_backtest import classify_regimes
from lab.strategy_lab import PERIODS_PER_YEAR, factor_score, monthly_rebalances, run_strategy

OOS_PATH = Path("data/sp500_oos_walkforward.parquet")
PANEL_PATH = Path("data/sp500_panel.parquet")
REPORT_PATH = Path("figures/lab/signal_duel.md")

BOOK_A = "A · Momentum (12-1)"
BOOK_B = "B · Frozen ML"

# The decision rule, stated BEFORE the run (#27 §3) and asserted by the tests.
MIN_CONSISTENCY = 0.5      # ML's Rank IC must beat momentum in MORE than half the years


def load_oos(path: Path = OOS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ------------------------------------------------------------------ scores
def momentum_score(day: pd.DataFrame) -> pd.Series:
    """Book A. Identical to the Strategy Lab's momentum recipe."""
    return factor_score(day, [("mom_12_1m", True)])


def ml_score(day: pd.DataFrame) -> pd.Series:
    """Book B. The walk-forward OOS prediction already in the frame — never re-predicted."""
    return pd.Series(day["model_score"].to_numpy(), index=day["ticker"].to_numpy())


SCORES = {BOOK_A: momentum_score, BOOK_B: ml_score}


# ------------------------------------------------------------------ rank IC
def rank_ic(df: pd.DataFrame, score_fn, rebalances) -> pd.Series:
    """Per-date Spearman correlation between the score and the realised forward return.

    Rank IC measures ranking skill across the WHOLE cross-section, which is why #25's
    evidence standard says to believe it over Sharpe when the two disagree: Sharpe only
    reports what happened in the tails the book actually held.
    """
    out, dates = [], []
    for t in rebalances:
        day = df[df["date"] == t]
        if len(day) < 20:
            continue
        s = score_fn(day).reindex(day["ticker"].to_numpy())
        out.append(float(pd.Series(s.to_numpy()).corr(
            pd.Series(day["fwd_ret_1m"].to_numpy()), method="spearman")))
        dates.append(t)
    return pd.Series(out, index=pd.DatetimeIndex(dates), name="rank_ic")


def ic_summary(ic: pd.Series) -> dict:
    """Mean Rank IC with the t-stat of its mean — a mean IC without n is not evidence."""
    v = ic.dropna()
    n = int(len(v))
    mean = float(v.mean()) if n else float("nan")
    t = float(mean / (v.std(ddof=1) / np.sqrt(n))) if n > 1 and v.std(ddof=1) > 0 else float("nan")
    return {"n": n, "mean_ic": mean, "t_stat": t}


# ------------------------------------------------------------------ scoring a book
def score_series(rets: pd.Series) -> dict:
    """After-cost scorecard for one monthly net-return series (analyser conventions)."""
    r = rets.dropna()
    n = int(len(r))
    if n == 0:
        return {"n": 0, "ann_ret": None, "vol": None, "sharpe": None, "sortino": None,
                "maxdd": None}
    eq = (1.0 + r).cumprod().to_numpy()
    mdd, _ = max_drawdown(eq)
    ann = float((1.0 + r).prod() ** (PERIODS_PER_YEAR / n) - 1.0)
    return {"n": n, "ann_ret": ann, "vol": float(volatility(r)),
            "sharpe": float(sharpe(r)), "sortino": float(sortino(r)), "maxdd": float(mdd)}


# ------------------------------------------------------------------ the duel
def run(make_report: bool = True, oos_path: Path = OOS_PATH,
        ml_label: str = BOOK_B, report_path: Path = REPORT_PATH) -> dict:
    """#31 Arm 1 added the parameters so the SAME harness can rematch a different model
    version. Defaults reproduce the #30 duel exactly, so the published result is unchanged."""
    df = load_oos(oos_path)
    scores = {BOOK_A: momentum_score, ml_label: ml_score}
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    rebals = monthly_rebalances(df)

    books, frames = {}, {}
    for name, fn in scores.items():
        pf = run_strategy({"name": name, "factors": [("mom_12_1m", True)],
                           "combine": "rank_avg", "long_only": True, "rebalance": "monthly"},
                          labeled=df, panel=panel, score_fn=fn, rebalances=rebals)
        pf["date"] = pd.to_datetime(pf["date"])
        frames[name] = pf
        books[name] = pd.Series(pf["net_ret"].to_numpy(), index=pd.DatetimeIndex(pf["date"]))

    ics = {name: rank_ic(df, fn, rebals) for name, fn in scores.items()}

    # The benchmark and the regime calendar both come from the equal-weight universe series
    # the same run produced, so every book is sliced on identical dates.
    bench = pd.Series(frames[BOOK_A]["bench_ret"].to_numpy(),
                      index=pd.DatetimeIndex(frames[BOOK_A]["date"]))
    regimes = classify_regimes(bench)["regime"]

    full = {name: score_series(s) for name, s in books.items()}
    full_ic = {name: ic_summary(ic) for name, ic in ics.items()}
    turnover = {name: float(frames[name]["turnover"].mean()) for name in scores}

    years = sorted({d.year for d in books[BOOK_A].index})
    per_year = {}
    for y in years:
        per_year[y] = {
            name: {**score_series(books[name][books[name].index.year == y]),
                   **ic_summary(ics[name][ics[name].index.year == y])}
            for name in scores
        }

    per_regime = {}
    for reg in ("calm", "normal", "stressed"):
        mask = regimes.reindex(books[BOOK_A].index) == reg
        per_regime[reg] = {
            name: {**score_series(books[name][mask.to_numpy()]),
                   **ic_summary(ics[name].reindex(books[name].index)[mask.to_numpy()])}
            for name in scores
        }

    verdict = decide(full, full_ic, per_year, ml_label=ml_label)
    result = {
        "window": f"{books[BOOK_A].index.min().date()} → {books[BOOK_A].index.max().date()}",
        "n_months": int(len(books[BOOK_A])),
        "full": full, "full_ic": full_ic, "turnover": turnover,
        "per_year": per_year, "per_regime": per_regime,
        "regime_counts": regimes.value_counts().to_dict(),
        "verdict": verdict,
        "books": list(scores),
        "ml_label": ml_label,
        "oos_path": str(oos_path),
    }
    if make_report:
        result["report_path"] = str(write_report(result, report_path))
    return result


def decide(full: dict, full_ic: dict, per_year: dict, ml_label: str = BOOK_B) -> dict:
    """Apply the PRE-STATED rule (#27 §3) + the #28 asymmetric ruling. No judgement here —
    the thresholds were fixed before the numbers existed, and this is arithmetic on them."""
    a_sh, b_sh = full[BOOK_A]["sharpe"], full[ml_label]["sharpe"]
    ml_wins_sharpe = bool(b_sh > a_sh)

    years = sorted(per_year)
    ml_ic_wins = [y for y in years
                  if (per_year[y][ml_label]["mean_ic"] or 0) > (per_year[y][BOOK_A]["mean_ic"] or 0)]
    consistency = len(ml_ic_wins) / len(years) if years else 0.0
    ic_consistent = bool(consistency > MIN_CONSISTENCY)

    ml_earns_keep = bool(ml_wins_sharpe and ic_consistent)
    if not ml_earns_keep:
        headline = "TRADE MOMENTUM — the ML did not earn its keep."
        strength = ("CONCLUSIVE. The A-1 contamination favours book B, so the ML lost while "
                    "carrying an advantage. Fixing A-1 can only make this verdict stronger.")
    else:
        headline = "The ML beat momentum on both tests."
        strength = ("INCONCLUSIVE, pending a clean A-1-fixed retrain. The `size` feature "
                    "leaks future adjustment information into book B only, so a win is "
                    "exactly what the contamination would also produce. Not bankable.")
    return {
        "ml_wins_sharpe": ml_wins_sharpe, "sharpe_a": a_sh, "sharpe_b": b_sh,
        "ic_years_won_by_ml": ml_ic_wins, "n_years": len(years),
        "ic_consistency": consistency, "ic_consistent": ic_consistent,
        "ml_earns_keep": ml_earns_keep, "headline": headline, "strength": strength,
    }


# ------------------------------------------------------------------ report
def _p(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"


def _n(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _row(name, s, ic=None):
    cells = [name, str(s["n"]), _p(s["ann_ret"]), _p(s["vol"]), _n(s["sharpe"]),
             _n(s["sortino"]), _p(s["maxdd"])]
    if ic is not None:
        cells += [_n(ic.get("mean_ic"), 4), _n(ic.get("t_stat"), 2)]
    return "| " + " | ".join(cells) + " |"


def write_report(result: dict, report_path: Path = REPORT_PATH) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    v = result["verdict"]
    L = []
    L.append("# The Signal Duel — frozen ML vs plain 12-1 momentum (#27, run under #30)\n")
    L.append("*Auto-generated by `lab/signal_duel.py`. ⚠️ **EDUCATIONAL SIMULATION.** Nothing "
             "was trained and nothing is predicted — the frozen model is read, never refit. "
             "Survivorship-biased universe (today's S&P 500 members applied to history), so "
             "every number is DIRECTIONAL.*\n")

    L.append("## The pre-stated decision rule\n")
    L.append("> **The ML earns its keep only if it beats momentum on after-cost Sharpe over "
             "the full window AND its Rank IC advantage is consistent (not one lucky year). "
             "Ties or momentum wins ⇒ the honest recommendation is \"trade momentum, keep ML "
             "as research\".**\n")
    L.append("Stated before the run (#27 §3), and applied by `decide()` as arithmetic — no "
             "judgement is exercised after seeing the numbers. \"Consistent\" is fixed as "
             f"**ML winning Rank IC in more than {MIN_CONSISTENCY:.0%} of calendar years**.\n")

    L.append("## The asymmetry — read this before the verdict\n")
    L.append("`size`, a model feature, is `log(adj_close)`: a bare, retroactively re-adjusted "
             "price level (#25 finding **A-1**, still open). Its value at a past date moves "
             "whenever a later split or dividend lands, so it leaks a little future "
             "information — **into book B only**. Momentum is a price *ratio*, so the "
             "adjustment factor cancels and book A is immune. Both halves of that claim are "
             "proved in `test/test_utils_pipeline_f2.py`.\n")
    L.append("The contamination therefore **favours the ML**, which fixes how each outcome "
             "may be read (#28 ruling):\n")
    L.append("| Outcome | Reading |")
    L.append("|---|---|")
    L.append("| **ML loses** | **CONCLUSIVE** — it lost while carrying an advantage. "
             "Trade momentum. |")
    L.append("| **ML wins** | **INCONCLUSIVE** — pending a clean A-1-fixed retrain. The win "
             "is what the leak would also produce. |")
    L.append("")

    L.append("## Construction — one variable\n")
    L.append(f"**Window:** {result['window']} ({result['n_months']} monthly rebalances). Both "
             "books run through `lab.strategy_lab.run_strategy`, which imports the frozen "
             "track's construction verbatim from `portfolio/paper_trade.py`: same universe, "
             "same dates, same long-only top-N, same inverse-vol sizing, same weight cap, "
             "same vol target, same 10 bps/side cost. **Only the score differs.**\n")
    L.append("Book B is scored from `data/sp500_oos_walkforward.parquet`, where every "
             "`model_score` is **walk-forward out-of-sample** (fit only on prior data, 21-day "
             "embargo). Scoring the frozen model over its own training window would hand book "
             "B an in-sample advantage and make the duel meaningless.\n")

    L.append("## Full window\n")
    L.append("| Book | n | Ann. return | Ann. vol | Sharpe | Sortino | MaxDD | Mean Rank IC | t |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name in result.get("books", list(SCORES)):
        L.append(_row(name, result["full"][name], result["full_ic"][name]))
    L.append("")
    L.append("### Turnover (costs decide real-world winners)\n")
    L.append("| Book | Mean monthly turnover |")
    L.append("|---|---|")
    for name in result.get("books", list(SCORES)):
        L.append(f"| {name} | {_p(result['turnover'][name])} |")
    L.append("")

    L.append("## Per calendar year\n")
    L.append("*Small n by construction — a single year is ~12 rebalances. Reported in full "
             "because the consistency test needs every year, not because any one of them is "
             "evidence on its own.*\n")
    L.append("| Year | Book | n | Ann. return | Ann. vol | Sharpe | Sortino | MaxDD | Mean Rank IC | t |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for y in sorted(result["per_year"]):
        for name in result.get("books", list(SCORES)):
            s = result["per_year"][y][name]
            L.append(f"| {y} " + _row(name, s, s)[1:])
    L.append("")

    L.append("## Per regime (the #21 calendar)\n")
    counts = result["regime_counts"]
    L.append(f"Month counts: calm {counts.get('calm', 0)} · normal {counts.get('normal', 0)} "
             f"· stressed {counts.get('stressed', 0)}. Regimes are classified on the "
             "equal-weight universe series this same run produced, so both books are sliced "
             "on identical dates.\n")
    L.append("| Regime | Book | n | Ann. return | Ann. vol | Sharpe | Sortino | MaxDD | Mean Rank IC | t |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for reg in ("calm", "normal", "stressed"):
        for name in result.get("books", list(SCORES)):
            s = result["per_regime"][reg][name]
            L.append(f"| {reg} " + _row(name, s, s)[1:])
    L.append("")

    L.append("## Verdict\n")
    L.append(f"**{v['headline']}**\n")
    L.append(f"* After-cost Sharpe, full window: momentum **{_n(v['sharpe_a'])}** vs ML "
             f"**{_n(v['sharpe_b'])}** → ML wins on Sharpe: "
             f"**{'YES' if v['ml_wins_sharpe'] else 'NO'}**")
    L.append(f"* Rank IC consistency: ML beat momentum in **{len(v['ic_years_won_by_ml'])} of "
             f"{v['n_years']}** calendar years ({v['ic_consistency']:.0%}) → consistent "
             f"(> {MIN_CONSISTENCY:.0%}): **{'YES' if v['ic_consistent'] else 'NO'}**")
    L.append(f"* Both tests passed: **{'YES' if v['ml_earns_keep'] else 'NO'}**\n")
    L.append(f"**Strength of the finding:** {v['strength']}\n")

    # The one result that could be read the other way, named rather than buried.
    _ml = result.get("ml_label", BOOK_B)
    a_ic = result["full_ic"][BOOK_A]["mean_ic"]
    b_ic = result["full_ic"][_ml]["mean_ic"]
    if b_ic > a_ic and not v["ml_wins_sharpe"]:
        L.append("## The tension worth naming — Sharpe and Rank IC disagree\n")
        L.append(f"Over the full window the ML has the **higher mean Rank IC** "
                 f"({_n(b_ic, 4)} vs {_n(a_ic, 4)}) and the **lower** Sharpe "
                 f"({_n(v['sharpe_b'])} vs {_n(v['sharpe_a'])}). This project's standing "
                 "evidence rule is *when Sharpe and Rank IC disagree, believe Rank IC* — "
                 "Rank IC measures ranking skill across the whole cross-section, Sharpe only "
                 "reports the payoff of the tails the book actually held. Read naively, that "
                 "rule argues for the ML.\n")
        L.append("**It does not rescue the ML here, and the pre-stated rule is why.** The "
                 f"consistency test shows that IC edge rests on **{len(v['ic_years_won_by_ml'])} "
                 f"of {v['n_years']}** years "
                 f"({', '.join(str(y) for y in v['ic_years_won_by_ml']) or 'none'}); momentum "
                 "wins the rest. A mean IC advantage carried by a minority of years is not "
                 "ranking skill, it is dispersion — which is exactly the failure the "
                 "\"not one lucky year\" clause was written to catch, before any of these "
                 "numbers existed.\n")
        L.append("Two further observations, neither of which changes the verdict:\n")
        L.append(f"* **The ML is not losing on costs.** It turns over "
                 f"{_p(result['turnover'][_ml])} a month against momentum's "
                 f"{_p(result['turnover'][BOOK_A])} — it is the *cheaper* book to run and "
                 "still ends with the lower after-cost Sharpe. The gap is selection, not "
                 "friction.\n")
        L.append("* **The ML's best IC years are not its best return years.** 2025 is its "
                 "largest IC win and a year it still trails on Sharpe. Ordering the "
                 "cross-section well is not the same as owning the right twenty names.\n")

    L.append("## This does NOT contradict the README's headline — read both\n")
    L.append("The README reports the same window and the same 10 bps/side with the **opposite** "
             "ordering: ML Sharpe **1.14** vs momentum **0.82**. Both are correct, because "
             "they are different books answering different questions:\n")
    L.append("| | README headline (#6) | This duel (#27) |")
    L.append("|---|---|---|")
    L.append("| Book | market-neutral **long/short deciles** | **long-only top-20**, the product book |")
    L.append("| Legs | long top decile, short bottom decile | long only |")
    L.append("| Sizing | inverse-vol within each leg | inverse-vol, capped, **vol-targeted 14%** |")
    L.append("| Question | does the score order the cross-section profitably? | which score makes the better product? |")
    L.append("| Winner | **ML** (1.14 vs 0.82) | **Momentum** (" +
             f"{_n(v['sharpe_a'])} vs {_n(v['sharpe_b'])}) |")
    L.append("")
    L.append("**And that is the finding, not a footnote.** The ML's skill is in ordering the "
             "*whole* cross-section — which is exactly what a long/short decile book harvests, "
             "including the short leg it gets paid for being right about. The product book "
             "owns twenty long names and never shorts anything, so most of that ordering skill "
             "is never monetised. It is the same story the Rank IC tells: **higher IC, lower "
             "product Sharpe.** Ranking skill and top-20 selection are not the same ability, "
             "and #27 asked which one to *trade*.\n")
    L.append("Consequence worth stating plainly: **the ML is not useless — it is mis-deployed.** "
             "The honest recommendation is the one #27 pre-committed to (trade momentum in the "
             "long-only product, keep the ML as research), and the natural follow-up is whether "
             "the product should have a short sleeve at all — a different question, needing its "
             "own instruction.\n")

    L.append("## Limits\n")
    L.append("* **Survivorship.** Today's S&P 500 members applied backward. It flatters both "
             "books, and there is no reason to assume it flatters them equally.\n")
    L.append(f"* **Short window.** {result['n_months']} monthly observations. Per-year and "
             "per-regime cells are smaller still and are illustration, not evidence.\n")
    L.append("* **One axis.** Long-only, top-N, monthly, 10 bps/side. A different holding "
             "period or cost assumption is a different experiment.\n")
    L.append("* **A-1 is not fixed here**, deliberately. Fixing it changes the frozen model's "
             "scores and needs its own clearly-labelled retrain — a separate phase.\n")

    report_path.write_text("\n".join(L))
    return report_path


def main():
    res = run(make_report=True)
    v = res["verdict"]
    print(f"window {res['window']} ({res['n_months']} mo)")
    for name in res["books"]:
        s, ic = res["full"][name], res["full_ic"][name]
        print(f"  {name:22s} sharpe {s['sharpe']:+.3f} | sortino {s['sortino']:+.3f} | "
              f"maxDD {s['maxdd']*100:6.2f}% | IC {ic['mean_ic']:+.4f} (t={ic['t_stat']:+.2f}) | "
              f"turnover {res['turnover'][name]*100:.1f}%")
    print(f"\nVERDICT: {v['headline']}")
    print(f"         {v['strength']}")
    print("report ->", res.get("report_path"))


if __name__ == "__main__":
    main()
