"""
Value factor — Phase 18. Built on the point-in-time fundamentals #17 cleared.

The composite (all four oriented so HIGHER = CHEAPER = better)
--------------------------------------------------------------
    earnings yield   = EPS(ttm)   / price
    book-to-market   = book value / market cap
    EBITDA/EV        = EBITDA(ttm)/ enterprise value
    FCF yield        = FCF(ttm)   / market cap

Each ratio is winsorized at the 1st/99th percentile, z-scored cross-sectionally on the
rebalance date, then averaged. A name needs at least `MIN_RATIOS` (2) of the four present;
below that its `value_score` is missing.

No ML. Equal-weight combine — exactly like the low-vol test in #14 — so any change in the
scorecard is attributable to the factor rather than to a refit.

Leakage control (the whole point of #17)
----------------------------------------
Every fundamental carries the EDGAR `filed` date. At a rebalance `t` we use only rows with
`publication_date <= t - PUBLICATION_LAG_DAYS`, so a filing that hit EDGAR on the rebalance
day itself cannot enter that day's book. Prices are the panel close at `t`; per-share facts
were put on the panel's split basis in `audit/sec_provider.py`. Fundamentals are the
EARLIEST-filed value for each period, never a later restatement.

⚠️ EDUCATIONAL SIMULATION. Survivorship caveat unchanged: the ticker list is today's S&P
500 membership, so results are DIRECTIONAL only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from audit.fundamentals import winsorize, zscore, WINSOR_PCT
from audit.sec_provider import SplitBasisUnavailable

logger = logging.getLogger("value_factor")

RATIOS = ["earnings_yield", "book_to_market", "ebitda_to_ev", "fcf_yield"]
MIN_RATIOS = 2                  # a name needs at least this many of the four
PUBLICATION_LAG_DAYS = 1        # a filing is tradable the day AFTER it hits EDGAR
FUND_PATH = Path("data/sec_fundamentals.parquet")


# ------------------------------------------------------------------ fundamentals table
def build_fundamentals(tickers, client=None, panel=None, out: Path | None = FUND_PATH,
                       progress_every: int = 50) -> pd.DataFrame:
    """Concatenate every ticker's point-in-time ledger into one tidy frame.

    Columns: ticker, period_end, publication_date, eps, book_value, ebitda,
    free_cash_flow, cash, debt_lt, debt_st, shares. Prices/market caps are deliberately NOT
    carried over — those are struck at the rebalance date, not at publication.
    """
    if client is None:
        from audit.sec_provider import SECClient
        client = SECClient(price_panel=panel)
    if getattr(client, "splits", None) is None:
        client.splits = client.fetch_splits(tickers)

    keep = ["ticker", "period_end", "publication_date", "eps", "book_value", "ebitda",
            "free_cash_flow", "cash", "debt_lt", "debt_st", "shares"]
    frames, missing = [], []
    for i, tk in enumerate(tickers):
        try:
            led = client.ledger(tk)
        except SplitBasisUnavailable:
            # Not a bad name — a bad BASIS, which is wrong for every name equally. Swallowing
            # it into `missing` would turn "all our ratios are wrong" into "no data" (#25 A-2).
            raise
        except Exception as e:                      # noqa: BLE001 — one bad name isn't fatal
            logger.warning("ledger failed for %s: %s", tk, e)
            missing.append(tk)
            continue
        if led.empty:
            missing.append(tk)
            continue
        frames.append(led[keep])
        if progress_every and (i + 1) % progress_every == 0:
            logger.info("fundamentals: %d/%d tickers", i + 1, len(tickers))

    if not frames:
        raise RuntimeError("no fundamentals retrieved — refusing to build a value factor "
                           "on an empty table")
    fund = pd.concat(frames, ignore_index=True)
    fund["publication_date"] = pd.to_datetime(fund["publication_date"])
    fund["period_end"] = pd.to_datetime(fund["period_end"])
    fund = fund.sort_values(["ticker", "publication_date", "period_end"])
    logger.info("fundamentals: %d rows, %d tickers (%d with no data)",
                len(fund), fund["ticker"].nunique(), len(missing))
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fund.to_parquet(out, index=False)
    return fund


def load_fundamentals(path: Path = FUND_PATH) -> pd.DataFrame:
    fund = pd.read_parquet(path)
    fund["publication_date"] = pd.to_datetime(fund["publication_date"])
    fund["period_end"] = pd.to_datetime(fund["period_end"])
    return fund


# ------------------------------------------------------------------ point-in-time join
def as_of(fund: pd.DataFrame, when, lag_days: int = PUBLICATION_LAG_DAYS) -> pd.DataFrame:
    """The latest filing per ticker published on or before `when - lag_days`.

    This is the leakage gate in one line: `publication_date <= when - lag`. Ties on
    publication date are broken by the later fiscal period, which is the fresher report.

    The comparison is `<=`, not `<`. Same-day filings are excluded by the lag, not by the
    operator, so with the default `PUBLICATION_LAG_DAYS = 1` the effect is "strictly before
    `when`" — but set the lag to 0 and same-day filings would be admitted. Described
    precisely here because the previous wording ("strictly before `when`") would have become
    an active lie the moment anyone changed that constant (#25 D-4).
    """
    cutoff = pd.Timestamp(when) - pd.Timedelta(days=lag_days)
    f = fund[fund["publication_date"] <= cutoff]
    if f.empty:
        return f
    f = f.sort_values(["publication_date", "period_end"])
    return f.groupby("ticker", sort=False).tail(1).set_index("ticker")


def ratios_on(fund: pd.DataFrame, prices: pd.Series, when,
              lag_days: int = PUBLICATION_LAG_DAYS) -> pd.DataFrame:
    """The four value ratios for one rebalance date.

    `prices` is a ticker-indexed close on `when` (split-adjusted, matching the share counts).
    Market cap and EV are struck with that price and the newest PUBLISHED balance sheet, so
    the valuation is current while the fundamentals are lagged — which is how a value factor
    is supposed to work.
    """
    f = as_of(fund, when, lag_days)
    if f.empty:
        return pd.DataFrame(columns=RATIOS)
    px = prices.reindex(f.index)
    mcap = f["shares"] * px
    mcap = mcap.where(mcap > 0)
    debt = f["debt_lt"].fillna(0.0) + f["debt_st"].fillna(0.0)
    ev = (mcap + debt - f["cash"].fillna(0.0)).where(mcap.notna())
    ev = ev.where(ev > 0)

    book = f["book_value"].where(f["book_value"] > 0)      # negative equity ⇒ P/B undefined
    out = pd.DataFrame({
        "earnings_yield": f["eps"] / px.where(px > 0),
        "book_to_market": book / mcap,
        "ebitda_to_ev": f["ebitda"] / ev,
        "fcf_yield": f["free_cash_flow"] / mcap,
    }, index=f.index)
    return out.replace([np.inf, -np.inf], np.nan)


def composite(ratios: pd.DataFrame, min_ratios: int = MIN_RATIOS,
              winsor_pct=WINSOR_PCT) -> pd.Series:
    """winsorize → z-score → average available. NaN where fewer than `min_ratios` present."""
    if ratios.empty:
        return pd.Series(dtype="float64")
    z = pd.DataFrame(index=ratios.index)
    for c in RATIOS:
        col = ratios[c] if c in ratios else pd.Series(np.nan, index=ratios.index)
        if col.notna().sum() < 3:            # too thin to winsorize/standardise meaningfully
            z[c] = np.nan
            continue
        z[c] = zscore(winsorize(col.to_numpy(dtype="float64"), winsor_pct))
    n = z.notna().sum(axis=1)
    score = z.mean(axis=1, skipna=True).where(n >= min_ratios)
    return score.rename("value_score")


# ------------------------------------------------------------------ panel attachment
def value_panel(fund: pd.DataFrame, panel: pd.DataFrame, dates,
                lag_days: int = PUBLICATION_LAG_DAYS) -> pd.DataFrame:
    """`value_score` (+ the raw ratios) for every (date, ticker) on `dates`.

    `close` is used rather than `adj_close`: share counts are split-adjusted but not
    dividend-adjusted, so pairing them with a dividend-adjusted price would inflate every
    yield by the cumulative dividend factor — worse for high payers, i.e. exactly the value
    names.
    """
    px_all = panel[["date", "ticker", "close"]].dropna()
    rows = []
    for t in pd.to_datetime(pd.Series(list(dates))).sort_values():
        prices = px_all.loc[px_all["date"] == t].set_index("ticker")["close"]
        if prices.empty:
            continue
        r = ratios_on(fund, prices, t, lag_days)
        if r.empty:
            continue
        r = r.assign(value_score=composite(r), date=t)
        rows.append(r.reset_index().rename(columns={"index": "ticker"}))
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "value_score", *RATIOS])
    out = pd.concat(rows, ignore_index=True)
    return out[["date", "ticker", "value_score", *RATIOS]]


def attach(labeled: pd.DataFrame, vpanel: pd.DataFrame,
           fill_missing: bool = True) -> pd.DataFrame:
    """Merge `value_score` onto the labeled panel.

    `fill_missing`: names with no usable fundamentals get 0.0 — the cross-sectional MEAN
    z-score, i.e. a neutral value opinion. Dropping them instead would change the universe
    between A and B and make the comparison dishonest; leaving them NaN would silently hand
    them their momentum rank a second time (the rank-average skips NaNs), which is a hidden
    momentum overweight. Neutral is the only option that leaves the universe intact.
    """
    out = labeled.merge(vpanel[["date", "ticker", "value_score"]], on=["date", "ticker"],
                        how="left")
    out["value_covered"] = out["value_score"].notna()
    if fill_missing:
        out["value_score"] = out["value_score"].fillna(0.0)
    return out


# ------------------------------------------------------------------ strategy specs
MOMENTUM = {
    "name": "A · Momentum",
    "factors": [("mom_12_1m", True)],
    "combine": "rank_avg", "long_only": True, "rebalance": "monthly",
}

MOMENTUM_VALUE = {
    "name": "B · Momentum + value",
    "factors": [("mom_12_1m", True), ("value_score", True)],
    "combine": "rank_avg", "long_only": True, "rebalance": "monthly",
}

VALUE_ONLY = {
    "name": "C · Value only",
    "factors": [("value_score", True)],
    "combine": "rank_avg", "long_only": True, "rebalance": "monthly",
}
