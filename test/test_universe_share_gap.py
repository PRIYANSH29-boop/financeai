"""
Phase 24 — the universe share-count gap.

`universe.shares_outstanding` reads SEC's XBRL **frames** API, which returns only facts that
carry no dimensions. Multi-class filers tag `dei:EntityCommonStockSharesOutstanding` per share
class, so Alphabet, Meta, Berkshire, Visa, Mastercard, Ford, Comcast and ~44 other S&P 500
members have NO undimensioned share count anywhere in the frame. The pre-#24 `candidates()`
inner-joined the counts and returned only survivors, so all of them vanished without a trace
and the shipped 1,200-name universe holds 451 of 503 S&P names instead of ~500.

These tests are all offline: the SEC client and the price provider are faked, because the
point is the *join and accounting behaviour*, which is where the names were lost.
"""

import numpy as np
import pandas as pd
import pytest

import universe as uni


class FakeSEC:
    """Minimal stand-in for SECClient: just the ticker→candidates map."""

    def __init__(self, rows):
        # rows: [(ticker, cik, name, exchange), …]
        self._rows = rows

    def cik_candidates(self):
        out = {}
        for tk, cik, name, exch in self._rows:
            out.setdefault(tk, []).append({"cik": cik, "name": name, "exchange": exch})
        return out


ROWS = [
    ("AAA", 1, "Single Class Inc", "NYSE"),        # has an SEC share count
    ("BBB", 2, "Multi Class Corp", "Nasdaq"),      # multi-class: no undimensioned count
    ("CCC", 3, "Also Multi Corp", "NYSE"),         # multi-class
    ("DDD", 4, "Off Exchange Ltd", "OTC"),         # filtered out by exchange
    ("EEE", 5, "Cboe Listed Inc", "CBOE"),         # #24 widened the allowlist to include this
]


@pytest.fixture
def frames_only_aaa(monkeypatch):
    """Only CIK 1 and 5 appear in the frames payload — the multi-class filers do not."""
    # Share counts are large enough that 'last_close' x shares clears the $2B floor, so a
    # name that goes missing below has gone missing on the JOIN, not on the cap filter.
    def fake_shares(client=None, frames=None, cache=None):
        return pd.DataFrame([{"cik": 1, "entity": "Single Class Inc", "shares": 800_000_000,
                              "as_of": "2026-03-31"},
                             {"cik": 5, "entity": "Cboe Listed Inc", "shares": 700_000_000,
                              "as_of": "2026-03-31"}])
    monkeypatch.setattr(uni, "shares_outstanding", fake_shares)


# ------------------------------------------------------------------ the accounting fix
def test_candidates_returns_the_gap_instead_of_swallowing_it(frames_only_aaa):
    have, gap = uni.candidates(FakeSEC(ROWS))
    assert sorted(have["ticker"]) == ["AAA", "EEE"]
    # the whole point of #24: the dropped multi-class names are RETURNED, not lost
    assert sorted(gap["ticker"]) == ["BBB", "CCC"]


def test_candidates_tags_the_share_count_source(frames_only_aaa):
    have, _ = uni.candidates(FakeSEC(ROWS))
    assert set(have["shares_source"]) == {"sec_xbrl_frames"}


def test_candidates_still_excludes_off_exchange_registrants(frames_only_aaa):
    have, gap = uni.candidates(FakeSEC(ROWS))
    assert "DDD" not in set(have["ticker"]) | set(gap["ticker"])


def test_cboe_is_an_accepted_exchange():
    """CBOE-listed S&P members were excluded for no methodological reason before #24."""
    assert "CBOE" in uni.EXCHANGES


# ------------------------------------------------------------------ the recovery path
@pytest.fixture
def fake_market(monkeypatch):
    """Price screen: every name liquid enough, at a price that clears $2B on its shares.

    Records its calls — the price screen is the expensive network stage, so recovering the
    gap must not cost a second pass over it.
    """
    calls = []

    def screen(tickers, lookback_days=90, batch=200):
        tickers = sorted(set(tickers))
        calls.append(tickers)
        return pd.DataFrame({"ticker": tickers,
                             "last_close": [50.0] * len(tickers),
                             "median_dollar_volume": [5e6] * len(tickers)})
    monkeypatch.setattr(uni, "market_screen", screen)
    return calls


def test_price_screen_runs_once_over_both_cohorts(frames_only_aaa, fake_market,
                                                 fake_fallback, tmp_path):
    uni.build_universe(client=FakeSEC(ROWS), out=tmp_path / "u.csv", max_names=None)
    assert len(fake_market) == 1, "the gap must not trigger a second price download"
    assert fake_market[0] == ["AAA", "BBB", "CCC", "EEE"]


@pytest.fixture
def fake_fallback(monkeypatch):
    calls = []

    def fb(tickers, batch=1):
        calls.append(sorted(tickers))
        return pd.DataFrame([{"ticker": tk, "shares": 900_000_000} for tk in sorted(tickers)])
    monkeypatch.setattr(uni, "fallback_shares", fb)
    return calls


def test_build_universe_recovers_the_multi_class_names(frames_only_aaa, fake_market,
                                                       fake_fallback, tmp_path):
    res = uni.build_universe(client=FakeSEC(ROWS), out=tmp_path / "u.csv",
                             max_names=None)
    tickers = set(res["universe"]["ticker"])
    assert {"BBB", "CCC"} <= tickers, "multi-class names still missing after the fix"
    assert {"AAA", "EEE"} <= tickers


def test_recovery_is_only_attempted_for_the_gap(frames_only_aaa, fake_market,
                                                fake_fallback, tmp_path):
    """The fallback costs one request per name, so it must not be asked about names that
    already have an SEC count."""
    uni.build_universe(client=FakeSEC(ROWS), out=tmp_path / "u.csv", max_names=None)
    assert fake_fallback == [["BBB", "CCC"]]


def test_recovered_names_are_tagged_as_provider_sourced(frames_only_aaa, fake_market,
                                                        fake_fallback, tmp_path):
    """An audit must be able to separate SEC-sourced market caps from provider-sourced ones."""
    res = uni.build_universe(client=FakeSEC(ROWS), out=tmp_path / "u.csv", max_names=None)
    src = res["universe"].set_index("ticker")["shares_source"]
    assert src["AAA"] == "sec_xbrl_frames"
    assert src["BBB"] == "price_provider"
    assert res["stats"]["shares_source_counts"] == {"sec_xbrl_frames": 2,
                                                    "price_provider": 2}


def test_stats_report_the_gap_and_the_recovery(frames_only_aaa, fake_market,
                                               fake_fallback, tmp_path):
    res = uni.build_universe(client=FakeSEC(ROWS), out=tmp_path / "u.csv", max_names=None)
    s = res["stats"]
    assert s["share_count_gap"] == 2
    assert s["share_count_recovered_from_price_provider"] == 2
    assert s["registrants_listed_with_shares"] == 2


def test_gap_is_persisted_next_to_the_universe(frames_only_aaa, fake_market,
                                              fake_fallback, tmp_path):
    """The missing artifact is what let this hide for four phases."""
    out = tmp_path / "u.csv"
    uni.build_universe(client=FakeSEC(ROWS), out=out, max_names=None)
    gap_file = out.with_name("universe_share_gap.csv")
    assert gap_file.exists()
    assert sorted(pd.read_csv(gap_file)["ticker"]) == ["BBB", "CCC"]


def test_recovery_can_be_switched_off(frames_only_aaa, fake_market, fake_fallback, tmp_path):
    res = uni.build_universe(client=FakeSEC(ROWS), out=tmp_path / "u.csv", max_names=None,
                             recover_share_gap=False)
    assert {"BBB", "CCC"}.isdisjoint(set(res["universe"]["ticker"]))
    assert fake_fallback == []


def test_illiquid_gap_names_do_not_cost_a_fallback_request(frames_only_aaa, monkeypatch,
                                                           fake_fallback, tmp_path):
    """Most of the ~2,300 names without an SEC count are penny/illiquid and would fail the
    $2B floor anyway; they must be pre-screened out before the per-name requests."""
    def screen(tickers, lookback_days=90, batch=200):
        tickers = sorted(set(tickers))
        return pd.DataFrame({
            "ticker": tickers,
            # BBB is a sub-$1 name, CCC is illiquid — neither deserves a request
            "last_close": [0.4 if t == "BBB" else 50.0 for t in tickers],
            "median_dollar_volume": [1e3 if t in ("BBB", "CCC") else 5e6 for t in tickers],
        })
    monkeypatch.setattr(uni, "market_screen", screen)
    uni.build_universe(client=FakeSEC(ROWS), out=tmp_path / "u.csv", max_names=None)
    assert fake_fallback == [[]] or fake_fallback == []


def test_fallback_rejects_unusable_share_counts(monkeypatch):
    """A provider returning None / 0 / NaN must drop the name, not create a $0 market cap."""
    class FakeTicker:
        def __init__(self, tk):
            self.tk = tk

        def get_info(self):
            return {"AAA": {"sharesOutstanding": 5_000_000},
                    "BBB": {"sharesOutstanding": None},
                    "CCC": {"sharesOutstanding": 0},
                    "DDD": {}}.get(self.tk, {})

    fake_yf = type("m", (), {"Ticker": FakeTicker})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    got = uni.fallback_shares(["AAA", "BBB", "CCC", "DDD"])
    assert list(got["ticker"]) == ["AAA"]
    assert got["shares"].iloc[0] == 5_000_000


def test_fallback_survives_a_provider_exception(monkeypatch):
    class Boom:
        def __init__(self, tk):
            self.tk = tk

        def get_info(self):
            if self.tk == "BBB":
                raise RuntimeError("rate limited")
            return {"sharesOutstanding": 1_000_000}

    fake_yf = type("m", (), {"Ticker": Boom})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    got = uni.fallback_shares(["AAA", "BBB"])
    assert list(got["ticker"]) == ["AAA"], "one bad name must not lose the whole batch"
