"""#28 A-2 — a missing split basis must fail LOUD, never warn-and-continue.

#25 finding A-2: `fetch_splits` logged a warning and returned `{}` when yfinance was
unavailable, so `ledger()` skipped the split adjustment and paired as-reported per-share
facts with split-adjusted prices. That is the NVDA-looks-10×-cheaper bug the #17 gate exists
to catch, reintroduced by a missing import, with no exception and every ratio wrong.

The invariant these tests pin: **an empty split map means "fetched, no splits exist"; it can
never mean "we could not find out".** The two used to be indistinguishable.
"""
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from audit.sec_provider import SECClient, SplitBasisUnavailable, split_factor  # noqa: E402


# ------------------------------------------------------------------ fetch_splits
def test_missing_yfinance_raises_instead_of_returning_empty(tmp_path, monkeypatch):
    """The exact A-2 scenario: yfinance not importable."""
    real_import = builtins.__import__

    def no_yfinance(name, *a, **kw):
        if name == "yfinance":
            raise ImportError("No module named 'yfinance'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_yfinance)
    with pytest.raises(SplitBasisUnavailable) as ei:
        SECClient.fetch_splits(["AAPL"], cache_dir=tmp_path)
    assert "yfinance" in str(ei.value)
    # The message has to say WHY it matters, or the next person deletes the raise.
    assert "split" in str(ei.value).lower()


def test_a_cached_split_map_is_still_served_without_a_download(tmp_path, monkeypatch):
    """A real cached basis is a real basis — the raise must not block offline reruns."""
    (tmp_path / "splits.json").write_text(json.dumps({"NVDA": [["2024-06-10", 10.0]]}))

    class ExplodingYF:
        @staticmethod
        def download(*a, **kw):
            raise AssertionError("must not download when the cache is present")

    monkeypatch.setitem(sys.modules, "yfinance", ExplodingYF)
    got = SECClient.fetch_splits(["NVDA"], cache_dir=tmp_path)
    assert got == {"NVDA": [["2024-06-10", 10.0]]}


def test_a_failed_chunk_raises_and_is_not_cached(tmp_path, monkeypatch):
    """A dropped chunk is 'we do not know', not 'these names never split'."""
    class FakeYF:
        @staticmethod
        def download(*a, **kw):
            raise RuntimeError("network down")

    monkeypatch.setitem(sys.modules, "yfinance", FakeYF)
    with pytest.raises(SplitBasisUnavailable) as ei:
        SECClient.fetch_splits(["AAPL", "MSFT"], cache_dir=tmp_path, batch=1)
    assert "chunk" in str(ei.value).lower()
    assert not (tmp_path / "splits.json").exists(), "a partial split map must never be cached"


# ------------------------------------------------------------------ ledger guard
def test_ledger_refuses_when_splits_were_never_fetched(tmp_path):
    """The second half of the fix: even if fetch_splits is bypassed, the ledger refuses."""
    c = SECClient(cache_dir=tmp_path)
    assert c.splits is None
    with pytest.raises(SplitBasisUnavailable) as ei:
        c.ledger("AAPL")
    assert "fetch_splits" in str(ei.value)      # tells the caller how to fix it


def test_empty_dict_is_an_accepted_deliberate_assertion(tmp_path, monkeypatch):
    """`splits = {}` means 'checked, nothing split'. That must still work — otherwise the
    fix would make a legitimately split-free universe unauditable."""
    c = SECClient(cache_dir=tmp_path)
    c.splits = {}
    monkeypatch.setattr(SECClient, "_build_ledger",
                        lambda self, ticker: pd.DataFrame())   # empty ledger, no network
    assert c.ledger("AAPL").empty                              # returns, does not raise


# ------------------------------------------------------------------ the maths it protects
def test_split_factor_is_what_the_raise_is_protecting():
    """Documents the size of the error the silent skip used to introduce."""
    nvda = [["2024-06-10", 10.0]]
    assert split_factor(nvda, "2024-01-31") == 10.0    # pre-split period: 10× adjustment
    assert split_factor(nvda, "2024-12-31") == 1.0     # post-split: none needed
    assert split_factor([], "2024-01-31") == 1.0       # genuinely no splits: neutral
