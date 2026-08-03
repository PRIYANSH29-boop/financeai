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


def _universe_was_built_with_the_exclusion() -> bool:
    """True only if the committed universe came out of the fixed builder.

    `excluded_non_equity` is written by `build_universe` whenever the exclusion runs, so its
    absence means the artifact predates the fix. As of 2026-08-03 it does: the rebuild is
    blocked because the price provider rate-limits the share-count recovery, and the guard
    (correctly) refuses to write a universe missing its multi-class names.
    """
    from pathlib import Path
    import json
    p = Path("data/universe_stats.json")
    return p.exists() and "excluded_non_equity" in json.loads(p.read_text())


@pytest.mark.skipif(not _universe_was_built_with_the_exclusion(),
                    reason="committed universe predates the non-equity exclusion — it still "
                           "contains ~21 commodity/crypto trusts. Rebuild with `make universe` "
                           "once the price provider stops rate-limiting.")
def test_the_built_universe_contains_no_non_equities():
    """End-to-end on the real artifact: after a rebuild the universe must hold ZERO funds/trusts.

    This asserts the post-fix invariant, not the old contaminated state — a universe built by
    the current code that still contained SPY would mean the exclusion never ran.
    """
    import json
    from pathlib import Path
    u = pd.read_csv("data/universe_midlarge.csv")
    cache = Path("data/cache/sector_sic_cache.json")
    sics = u["cik"].astype(str).map(json.loads(cache.read_text())) if cache.exists() else None
    survivors = sorted(u.loc[uni.is_non_equity(u["name"], sics), "ticker"])
    assert not survivors, f"non-equities still in the universe: {survivors}"
    assert not ({"SPY", "QQQ", "DIA", "MDY", "GLD", "SLV", "IBIT", "GBTC"} & set(u["ticker"]))


@pytest.mark.skipif(not __import__("pathlib").Path("data/universe_midlarge.csv").exists(),
                    reason="committed universe not built")
def test_real_companies_that_look_like_trusts_survived_the_build():
    """The over-reach direction, on the real artifact: REITs and banks must still be present."""
    u = set(pd.read_csv("data/universe_midlarge.csv")["ticker"])
    for tk in ["CPT", "DLR", "ESS", "FRT", "NTRS", "NFLX"]:
        assert tk in u, f"{tk} was wrongly excluded from the universe"


# ─────────────────────────────────────────── degraded-recovery guard (2026-08-03 regression)
def test_build_universe_refuses_a_degraded_recovery(monkeypatch):
    """The real failure: yfinance rate-limited mid-run, `fallback_shares` swallowed all 852
    exceptions at debug level, and the build wrote a universe silently missing every
    multi-class mega-cap — S&P coverage back to 451/503 with no error anywhere."""
    cand = pd.DataFrame({"ticker": ["AAA"], "cik": [1], "name": ["AAA CORP"],
                         "exchange": ["NYSE"], "shares": [1e9], "as_of": ["2026-01-01"],
                         "shares_source": ["sec_xbrl_frames"]})
    gap = pd.DataFrame({"ticker": ["GOOGL"], "cik": [2], "name": ["Alphabet Inc."],
                        "exchange": ["Nasdaq"]})
    scr = pd.DataFrame({"ticker": ["AAA", "GOOGL"], "last_close": [100.0, 200.0],
                        "median_dollar_volume": [1e9, 1e9]})
    monkeypatch.setattr(uni, "candidates", lambda client=None: (cand, gap))
    monkeypatch.setattr(uni, "market_screen", lambda *a, **k: scr)
    monkeypatch.setattr(uni, "fallback_shares",
                        lambda tickers, batch=1: pd.DataFrame(columns=["ticker", "shares"]))

    with pytest.raises(RuntimeError, match="recovery degraded"):
        uni.build_universe(out=None)


def test_the_guard_can_be_overridden_deliberately():
    """An explicit opt-out exists, so the guard never blocks a knowing operator."""
    import inspect
    assert "fail_on_degraded_recovery" in inspect.signature(uni.build_universe).parameters


def test_a_healthy_recovery_passes_the_guard(monkeypatch):
    cand = pd.DataFrame({"ticker": ["AAA"], "cik": [1], "name": ["AAA CORP"],
                         "exchange": ["NYSE"], "shares": [1e9], "as_of": ["2026-01-01"],
                         "shares_source": ["sec_xbrl_frames"]})
    gap = pd.DataFrame({"ticker": ["GOOGL"], "cik": [2], "name": ["Alphabet Inc."],
                        "exchange": ["Nasdaq"]})
    scr = pd.DataFrame({"ticker": ["AAA", "GOOGL"], "last_close": [100.0, 200.0],
                        "median_dollar_volume": [1e9, 1e9]})
    monkeypatch.setattr(uni, "candidates", lambda client=None: (cand, gap))
    monkeypatch.setattr(uni, "market_screen", lambda *a, **k: scr)
    monkeypatch.setattr(uni, "fallback_shares", lambda tickers, batch=1: pd.DataFrame(
        {"ticker": ["GOOGL"], "shares": [5.8e9]}))
    monkeypatch.setattr(uni, "sic_codes", lambda ciks, **k: pd.Series(dtype=object))
    res = uni.build_universe(out=None)
    assert set(res["universe"]["ticker"]) == {"AAA", "GOOGL"}
    assert res["stats"]["share_count_recovered_from_price_provider"] == 1


# ────────────────────────────────────── fallback_shares pacing/backoff/cache (2026-08-03 fix)
def _yf_stub(answers, calls):
    """Fake yfinance module whose Ticker(tk).get_info() follows `answers`."""
    class _T:
        def __init__(self, tk): self.tk = tk
        def get_info(self):
            calls.append(self.tk)
            v = answers[self.tk]
            if isinstance(v, Exception):
                raise v
            return {"sharesOutstanding": v}
    return type("yf", (), {"Ticker": _T})


def _install(monkeypatch, answers, calls):
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", _yf_stub(answers, calls))
    monkeypatch.setattr("time.sleep", lambda *_: None)      # no real waiting in tests


def test_fallback_shares_caches_results_so_a_rerun_resumes(monkeypatch, tmp_path):
    """A 20-minute loop that dies at 90% used to lose everything. Now progress is on disk."""
    calls = []
    _install(monkeypatch, {"AAA": 1e9, "BBB": 2e9}, calls)
    c = tmp_path / "shares.json"
    r1 = uni.fallback_shares(["AAA", "BBB"], cache_path=c, cool_down=0)
    assert len(r1) == 2 and c.exists()
    n_first = len(calls)
    r2 = uni.fallback_shares(["AAA", "BBB"], cache_path=c, cool_down=0)
    assert len(r2) == 2
    assert len(calls) == n_first, "second run refetched instead of using the cache"


def test_fallback_shares_caches_misses_too(monkeypatch, tmp_path):
    """A miss must be remembered, or every re-run pays for the same dead ticker again."""
    calls = []
    _install(monkeypatch, {"DEAD": None}, calls)
    c = tmp_path / "shares.json"
    uni.fallback_shares(["DEAD"], cache_path=c, cool_down=0, max_retries=1)
    import json as _j
    assert _j.loads(c.read_text()) == {"DEAD": None}


def test_fallback_shares_retries_with_backoff_before_giving_up(monkeypatch, tmp_path):
    """Two transient errors then a success must still recover the name."""
    calls = []
    state = {"n": 0}

    class _T:
        def __init__(self, tk): pass
        def get_info(self):
            calls.append(1)
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("429 rate limited")
            return {"sharesOutstanding": 5.8e9}
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", type("yf", (), {"Ticker": _T}))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    out = uni.fallback_shares(["GOOGL"], cache_path=tmp_path / "s.json", cool_down=0,
                              max_retries=3)
    assert len(out) == 1 and out.iloc[0]["shares"] == 5.8e9
    assert len(calls) == 3, "did not retry"


def test_fallback_shares_trips_a_circuit_breaker_instead_of_grinding(monkeypatch, tmp_path):
    """The real 2026-08-03 failure: 853 straight refusals. Stop early, don't burn 20 minutes."""
    calls = []
    tickers = [f"T{i}" for i in range(200)]
    _install(monkeypatch, {t: RuntimeError("429") for t in tickers}, calls)
    out = uni.fallback_shares(tickers, cache_path=tmp_path / "s.json", cool_down=0,
                              max_retries=1, circuit_break=10)
    assert len(out) == 0
    assert len(calls) < 50, f"kept going after the breaker should have tripped ({len(calls)} calls)"


def test_fallback_shares_does_not_use_the_aggregate_share_count():
    """`fast_info.shares` is cheaper but reports A+B+C (GOOGL 12.2B) against a Class-A price,
    which would roughly double the market cap. The source must stay `get_info`."""
    import inspect
    src = inspect.getsource(uni.fallback_shares)
    assert "get_info" in src
    assert "fast_info" not in src.split('"""')[2], "fast_info must not be used in the body"


# ──────────────────────────────────────── exporter-side filter (live-site hotfix, 2026-08-03)
def test_the_shipped_explore_bundle_contains_no_non_equities():
    """Explore is a STOCK explorer. GLD/SLV/IBIT/GBTC were rendering as rows on the live site
    for months — never basket-eligible, so untradable, but shown as if they were companies.

    This asserts the shipped artifact directly, so it holds regardless of which universe the
    bundle was exported from.
    """
    import json
    from pathlib import Path
    p = Path("web/public/bundle/explore.json")
    if not p.exists():
        pytest.skip("bundle not built")
    rows = json.loads(p.read_text())["rows"]
    names = pd.Series([r["ticker"] for r in rows])
    known = {"GLD", "SLV", "IBIT", "GBTC", "PHYS", "IAU", "SGOL", "BITB", "BTC", "ETHA",
             "SPY", "QQQ", "DIA", "MDY"}
    present = sorted(known & set(names))
    assert not present, f"non-equities still rendering in Explore: {present}"


def test_the_exporter_filters_non_equities_itself():
    """Belt-and-braces: the exporter must not rely on the universe artifact already being
    clean, because a bundle can be exported from a universe built before the builder fix."""
    import inspect
    import scripts.export_web_bundle as ex
    src = inspect.getsource(ex.build_explore)
    assert "is_non_equity" in src, "exporter no longer filters non-equities"
