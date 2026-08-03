"""The non-equity exclusion (#26 follow-up).

#24's share-gap recovery reopened the universe to things that are not companies: an ETF or
commodity trust has no share count in the SEC frames endpoint, so it landed in the gap, the
price provider answered `sharesOutstanding` for it, and it cleared the $2B + liquidity screens
trivially. SPY entered as the single largest name in the 1,200.

The danger in fixing it is over-reach: REITs and banks are legally trusts too, and a bare "ETF"
substring matches N-ETF-LIX. These tests pin BOTH directions on the real entity names and SIC
codes that were actually in the shipped universe.
"""

import pandas as pd
import pytest

import universe as uni


# The real names + SIC codes observed in data/universe_midlarge.csv (the shipped 1,200).
NON_EQUITIES = [
    ("SPY", "SPDR S&P 500 ETF TRUST", None),
    ("QQQ", "INVESCO QQQ TRUST, SERIES 1", None),
    ("DIA", "SPDR DOW JONES INDUSTRIAL AVERAGE ETF TRUST", None),
    ("MDY", "SPDR S&P MIDCAP 400 ETF TRUST", None),
    ("IBIT", "iShares Bitcoin Trust ETF", "6221"),
    ("GBTC", "Grayscale Bitcoin Trust ETF", "6221"),
    ("ETHA", "iShares Ethereum Trust ETF", "6221"),
    ("SGOL", "abrdn Gold ETF Trust", "6221"),
    ("GLD", "SPDR GOLD TRUST", "6221"),
    ("IAU", "ISHARES GOLD TRUST", "6221"),
    ("SLV", "iShares Silver Trust", "6221"),
    ("PHYS", "Sprott Physical Gold Trust", "6221"),
    ("PSLV", "Sprott Physical Silver Trust", "6221"),
    ("GLDM", "World Gold Trust", "6221"),
    ("IAUM", "iShares Gold Trust Micro", "6221"),
    ("FBTC", "Fidelity Wise Origin Bitcoin Fund", "6221"),
    ("AGQ", "ProShares Trust II", "6221"),
    ("UVXY", "ProShares Trust II", "6221"),
    ("UVIX", "VS Trust", "6221"),
]

REAL_COMPANIES = [
    ("CPT", "CAMDEN PROPERTY TRUST", "6798"),          # REIT — legally a trust, a real company
    ("DLR", "DIGITAL REALTY TRUST, INC.", "6798"),
    ("ESS", "ESSEX PROPERTY TRUST, INC.", "6798"),
    ("FRT", "FEDERAL REALTY INVESTMENT TRUST", "6798"),
    ("HR", "Healthcare Realty Trust Inc", "6798"),
    ("NSA", "National Storage Affiliates Trust", "6798"),
    ("CTRE", "CareTrust REIT, Inc.", "6798"),
    ("NTRS", "NORTHERN TRUST CORP", "6022"),           # a bank named Trust
    ("WTFC", "WINTRUST FINANCIAL CORP", "6022"),
    ("HBAN", "HUNTINGTON BANCSHARES INC /MD/", "6021"),
    ("NFLX", "NETFLIX INC", "7841"),                   # the substring trap: N-ETF-LIX
    ("AAPL", "Apple Inc.", "3571"),
]


def _frame(rows):
    return pd.DataFrame(rows, columns=["ticker", "name", "sic"]).set_index("ticker")


def test_every_known_non_equity_is_excluded():
    f = _frame(NON_EQUITIES)
    hit = uni.is_non_equity(f["name"], f["sic"])
    missed = sorted(f.index[~hit])
    assert not missed, f"these funds/trusts survived the filter: {missed}"


def test_no_real_company_is_excluded():
    f = _frame(REAL_COMPANIES)
    hit = uni.is_non_equity(f["name"], f["sic"])
    caught = sorted(f.index[hit])
    assert not caught, f"these real companies were wrongly excluded: {caught}"


def test_netflix_survives_the_bare_etf_substring():
    """`\\bETF\\b` is word-bounded on purpose — 'NETFLIX' contains 'ETF'."""
    f = _frame([("NFLX", "NETFLIX INC", "7841")])
    assert not uni.is_non_equity(f["name"], f["sic"]).iloc[0]
    assert uni.is_non_equity(pd.Series(["SPDR S&P 500 ETF TRUST"])).iloc[0]


def test_the_sic_less_unit_investment_trusts_need_the_name_rule():
    """SPY/QQQ/DIA/MDY carry NO SIC, so structure alone cannot see them."""
    f = _frame([r for r in NON_EQUITIES if r[2] is None])
    assert uni.is_non_equity(f["name"], f["sic"]).all()
    # …and with SIC dropped entirely they must still be caught.
    assert uni.is_non_equity(f["name"], None).all()


def test_the_commodity_trusts_need_the_sic_rule():
    """GLD/IAU/SLV/PHYS/PSLV/GLDM/IAUM/FBTC/UVIX carry no ETF-ish name — SIC 6221 is what
    catches them. Name-only would silently keep every one."""
    sic_only = [r for r in NON_EQUITIES
                if r[2] == "6221" and not pd.Series([r[1]]).str.contains(
                    uni.NON_EQUITY_NAME_RE, case=False, regex=True).iloc[0]]
    assert len(sic_only) >= 8, "fixture no longer exercises the SIC-only path"
    f = _frame(sic_only)
    assert uni.is_non_equity(f["name"], f["sic"]).all()
    assert not uni.is_non_equity(f["name"], None).any()   # name rule alone would miss all of them


def test_sic_set_excludes_the_fund_codes_and_nothing_else():
    assert uni.NON_EQUITY_SIC == {"6221", "6722", "6726"}
    assert "6798" not in uni.NON_EQUITY_SIC      # REITs
    assert "6021" not in uni.NON_EQUITY_SIC      # banks
    assert "6022" not in uni.NON_EQUITY_SIC


def test_missing_sic_is_not_treated_as_a_fund():
    """A company whose SIC lookup failed must not be deleted for it."""
    f = _frame([("ACME", "ACME MANUFACTURING CORP", None)])
    assert not uni.is_non_equity(f["name"], f["sic"]).iloc[0]


def test_exclusion_runs_before_the_liquidity_cap():
    """Freed slots must refill with real companies, not shrink the universe — so the exclusion
    has to happen before `head(max_names)`. Pinned by reading the source order."""
    import inspect
    src = inspect.getsource(uni.build_universe)
    assert src.index("exclude_non_equity") < src.index("liquidity_cap_dropped")


@pytest.mark.skipif(not __import__("pathlib").Path("data/universe_midlarge.csv").exists(),
                    reason="committed universe not built")
def test_against_the_shipped_universe():
    """End-to-end on the real file: the 25 known non-equities go, every REIT and bank stays."""
    import json
    from pathlib import Path
    u = pd.read_csv("data/universe_midlarge.csv")
    cache = Path("data/cache/sector_sic_cache.json")
    sics = u["cik"].astype(str).map(json.loads(cache.read_text())) if cache.exists() else None
    hit = uni.is_non_equity(u["name"], sics)
    excluded = set(u.loc[hit, "ticker"])
    assert {"SPY", "QQQ", "DIA", "MDY", "GLD", "SLV", "IBIT", "GBTC"} <= excluded
    assert not ({"CPT", "DLR", "ESS", "FRT", "HR", "NSA", "NTRS", "WTFC", "NFLX"} & excluded)
