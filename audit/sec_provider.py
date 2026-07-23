"""
SEC EDGAR XBRL fundamentals provider — the point-in-time data source for #17/#18.

Why SEC and not FMP
-------------------
#17 was specified against Financial Modeling Prep because its statement endpoints carry a
`filingDate`. FMP is unreachable from this machine (connection refused, verified with the
sandbox off) — but `data.sec.gov` is, and it is the *source* FMP resells. Every XBRL fact
returned by the `companyfacts` API carries:

    {"start": ..., "end": <period end>, "val": ..., "form": "10-Q",
     "filed": "<the date the filing hit EDGAR>", "accn": ..., "fy": ..., "fp": ...}

`filed` is the real publication date — exactly what the point-in-time gate (Check 3) needs,
straight from the primary source with no vendor in between. Fetching is free, keyless, and
rate-limited to ~10 req/s with a declared User-Agent.

What this module produces
-------------------------
`SECClient.statements(ticker)` returns the SAME record shape `FMPClient.statements` does, so
`audit.fundamentals.run_audit` consumes either without change:

    {ticker, period_end, publication_date, period, reported_currency,
     revenue, eps, ebitda, book_value, free_cash_flow, enterprise_value, market_cap, price}

Modelling choices (all documented, none silent)
-----------------------------------------------
* **Flows are trailing-twelve-month.** Value ratios (E/P, EBITDA/EV, FCF yield) are TTM by
  convention. Quarterly facts come either directly (`80 ≤ end-start ≤ 100` days) or are
  *derived* from the year-to-date cumulative facts that share a fiscal-year `start`
  (ytd[i] − ytd[i−1]) — this is what recovers Q4, which filers report only inside the
  annual figure. TTM needs 4 consecutive quarters spanning 330–400 days, else NaN.
* **EPS TTM sums quarterly diluted EPS.** Not exactly equal to TTM net income / current
  share count when the share count moves; standard practice, and flagged in the report.
* **Restatements are ignored on purpose.** For each period we keep the EARLIEST `filed`
  value — the number that was actually public at the time. Using a later restatement would
  be look-ahead.
* **Stocks (equity, cash, debt, shares) are instantaneous** facts at the period end.
* **Price is the local panel's close on the first trading day ≥ publication date**, so
  market cap and EV are struck with information that existed at publication. SEC has no
  prices; this keeps the whole record point-in-time consistent.
* **EV = market cap + total debt − cash.** Minority interest and preferred are omitted
  (rarely tagged consistently); documented as an approximation.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("fund_audit.sec")

SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_CACHE = Path("data/sec_cache")          # slim extractions only, never the 4 MB raw JSON

# SEC asks for a descriptive User-Agent with contact details; without one it returns 403.
USER_AGENT = "RankAlpha educational research (priyansh2005p@gmail.com)"

# Only periodic reports — 8-K/S-1 exhibits restate odd windows and muddy the PIT picture.
ACCEPTED_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"}

QUARTER_DAYS = (80, 100)      # a "quarter" duration fact
TTM_DAYS = (330, 400)         # 4 quarters must span roughly a year

# concept name -> (kind, [xbrl tags in preference order])
CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("duration", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense", "InterestAndDividendIncomeOperating",  # banks
    ]),
    "eps": ("duration", [
        "EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic",
    ]),
    "operating_income": ("duration", [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ]),
    "dep_amort": ("duration", [
        "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization", "DepreciationNonproduction",
    ]),
    "cfo": ("duration", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]),
    "capex": ("duration", [
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
    ]),
    "book_value": ("instant", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    "cash": ("instant", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ]),
    "debt_lt": ("instant", ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"]),
    "debt_st": ("instant", ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings",
                            "OtherShortTermBorrowings", "CommercialPaper"]),
    "shares": ("instant", ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]),
    # Fallback share count for filers that never tag an undimensioned point-in-time count.
    # A weighted-average count is a LEVEL, not a flow, so it is never TTM-summed.
    "shares_wavg": ("level", [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ]),
}

# Tickers whose SEC ticker-file entry points at a registrant with no XBRL history — almost
# always a holdco reorganisation that left a fresh CIK holding the symbol while the entire
# filing history stayed on the predecessor. Verified by hand, one line each.
TICKER_CIK_OVERRIDES = {
    "XOM": 34088,        # ticker file → "ExxonMobil Holdings Corp" (no facts); history on Exxon Mobil Corp
}

FLOW_COLS = ["revenue", "eps", "operating_income", "dep_amort", "cfo", "capex"]
STOCK_COLS = ["book_value", "cash", "debt_lt", "debt_st", "shares"]


# ============================================================ pure logic (unit-tested)
def _dedupe_earliest_filed(facts: list[dict], key) -> list[dict]:
    """Collapse duplicates: for each period key, keep the most-preferred tag and, within a
    tag, the EARLIEST `filed`.

    Earliest-filed is the value the market actually had on publication day; a later
    restatement of the same period is information from the future and must not enter a
    point-in-time factor. `rank` (0 = most preferred tag) breaks ties first, so an era where
    a filer tagged the same number two ways resolves deterministically.
    """
    best: dict = {}
    for f in facts:
        k = key(f)
        prev = best.get(k)
        if prev is None or (f.get("rank", 0), f["filed"]) < (prev.get("rank", 0), prev["filed"]):
            best[k] = f
    return sorted(best.values(), key=lambda f: (f["end"], f["filed"]))


def quarterly_from_duration_facts(facts: list[dict]) -> list[dict]:
    """Turn raw duration facts into one clean quarterly series.

    Two sources, in priority order:
      1. **Direct** — facts whose span is a quarter (80–100 days).
      2. **Derived** — filers report year-to-date cumulatives sharing one fiscal-year
         `start` (3m, 6m, 9m, 12m). Consecutive differences recover the missing quarters,
         which is the only way to get Q4 out of a 10-K. The derived quarter inherits the
         `filed` date of the LATER cumulative — the date that information became public.

    Returns records {end, filed, val, source} sorted by period end, one per period.
    """
    facts = [f for f in facts if f.get("start") and f.get("end") and f.get("filed")]
    facts = _dedupe_earliest_filed(facts, key=lambda f: (f["start"], f["end"]))

    def days(f):
        return (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days

    out: dict[str, dict] = {}
    for f in facts:                                     # 1. direct quarters
        if QUARTER_DAYS[0] <= days(f) <= QUARTER_DAYS[1]:
            out[f["end"]] = {"end": f["end"], "filed": f["filed"], "val": f["val"],
                             "source": "direct"}

    by_start: dict[str, list[dict]] = {}
    for f in facts:                                     # 2. YTD differences
        by_start.setdefault(f["start"], []).append(f)
    for _start, group in by_start.items():
        group = sorted(group, key=lambda f: f["end"])
        for prev, cur in zip(group, group[1:]):
            span = (pd.Timestamp(cur["end"]) - pd.Timestamp(prev["end"])).days
            if not (QUARTER_DAYS[0] <= span <= QUARTER_DAYS[1]):
                continue
            if cur["end"] in out:                       # direct fact wins
                continue
            out[cur["end"]] = {
                "end": cur["end"],
                "filed": max(cur["filed"], prev["filed"]),
                "val": cur["val"] - prev["val"],
                "source": "derived_ytd",
            }
    return sorted(out.values(), key=lambda r: r["end"])


def ttm(series: pd.DataFrame, col: str = "val") -> pd.Series:
    """Trailing-twelve-month sum of a quarterly series, gap-aware.

    `series` is indexed by period end (sorted). A TTM value is emitted only where the four
    quarters in the window actually span 330–400 days — so a filer that skipped a quarter
    gets NaN instead of a silently short "year".
    """
    if series.empty:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(series.index)
    vals = series[col].to_numpy(dtype="float64")
    out = np.full(len(vals), np.nan)
    for i in range(3, len(vals)):
        # idx[i-3] and idx[i] are the ENDS of the oldest and newest quarter in the window,
        # so they are three quarters apart (~273 days) when no quarter is missing.
        span = (idx[i] - idx[i - 3]).days
        if not (TTM_DAYS[0] - 90 <= span <= TTM_DAYS[1] - 100):
            continue
        window = vals[i - 3:i + 1]
        if np.isnan(window).any():
            continue
        out[i] = window.sum()
    return pd.Series(out, index=series.index)


def split_factor(splits: list, when) -> float:
    """Cumulative split ratio applied AFTER `when` — the number that puts an as-reported
    per-share quantity onto today's basis.

    Why this is not optional: XBRL facts are as-reported (we deliberately keep the
    earliest-filed value, so they are never retro-adjusted), while the yfinance price panel
    rewrites history onto today's split basis. Pairing the two raw would have made NVDA's
    pre-June-2024 earnings yield 10× too high and ranked it as the cheapest name in the
    index. `shares × factor` and `eps ÷ factor` put both sides on the panel's basis.

    `splits` is [(iso_date, ratio), …]; ratios strictly after `when` are multiplied.
    """
    if not splits:
        return 1.0
    t = pd.Timestamp(when)
    f = 1.0
    for d, ratio in splits:
        if pd.Timestamp(d) > t and ratio and ratio > 0:
            f *= float(ratio)
    return f


def enterprise_value(market_cap, debt_lt, debt_st, cash) -> float:
    """EV = market cap + total debt − cash. Missing debt/cash legs count as zero (a
    documented approximation); a missing market cap makes EV undefined."""
    mc = _f(market_cap)
    if mc is None or mc <= 0:
        return float("nan")
    debt = (_f(debt_lt) or 0.0) + (_f(debt_st) or 0.0)
    return mc + debt - (_f(cash) or 0.0)


def _f(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# ============================================================ network client
@dataclass
class SECClient:
    """EDGAR XBRL client. Caches the *slim* per-ticker extraction (a few KB) rather than the
    multi-megabyte raw companyfacts blob, so re-runs are instant and the repo stays small.

    Never fabricates: a ticker with no CIK or no usable facts yields `[]` and is counted as
    a coverage miss in the report.
    """

    cache_dir: Path = SEC_CACHE
    user_agent: str = USER_AGENT
    min_interval: float = 0.12          # SEC fair-access: ≤10 requests/second
    price_panel: pd.DataFrame | None = None     # date, ticker, adj_close (for PIT prices)
    splits: dict | None = None                  # {ticker: [(date, ratio), …]}, see split_factor
    _last: float = field(default=0.0, repr=False)
    _cik: dict | None = field(default=None, repr=False)
    _px: dict | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- http
    def _get_json(self, url: str) -> dict:
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent, "Accept-Encoding": "gzip"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                self._last = time.monotonic()
                return json.loads(raw)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return {}

    # ---------------------------------------------------------------- ticker -> CIK
    def cik_candidates(self) -> dict[str, list[dict]]:
        """{TICKER: [{cik, name, exchange}, …]} from SEC's official ticker file.

        A ticker can map to more than one CIK — a holdco reorganisation leaves the new shell
        registered under the same symbol with no XBRL history (ExxonMobil Holdings vs Exxon
        Mobil Corp). We keep every candidate and let `_build_ledger` pick the one that
        actually has facts, instead of silently returning an empty ledger.
        """
        if self._cik is not None:
            return self._cik
        cf = Path(self.cache_dir) / "company_tickers_exchange.json"
        if cf.exists():
            payload = json.loads(cf.read_text())
        else:
            payload = self._get_json(SEC_TICKERS)
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps(payload))
        fields = payload["fields"]
        i_cik, i_name = fields.index("cik"), fields.index("name")
        i_tk, i_ex = fields.index("ticker"), fields.index("exchange")
        out: dict[str, list[dict]] = {}
        for row in payload["data"]:
            tk = str(row[i_tk] or "").upper().strip()
            if not tk:
                continue
            # yfinance spells class shares with '-', SEC with '.' (BRK.B vs BRK-B).
            out.setdefault(tk.replace(".", "-"), []).append(
                {"cik": int(row[i_cik]), "name": row[i_name], "exchange": row[i_ex]})
        self._cik = out
        return out

    def cik_map(self) -> dict[str, dict]:
        """{TICKER: {cik, name, exchange}} — the primary (first-listed) CIK per ticker."""
        return {tk: cands[0] for tk, cands in self.cik_candidates().items()}

    # ---------------------------------------------------------------- price lookup
    def _price_lookup(self) -> dict:
        """{ticker: (sorted dates, adj_close array)} for as-of price lookup."""
        if self._px is not None:
            return self._px
        px: dict = {}
        if self.price_panel is not None:
            p = self.price_panel[["date", "ticker", "adj_close"]].dropna()
            p = p.sort_values("date")
            for tk, g in p.groupby("ticker", sort=False):
                px[tk] = (g["date"].to_numpy(dtype="datetime64[ns]"),
                          g["adj_close"].to_numpy(dtype="float64"))
        self._px = px
        return px

    def price_on(self, ticker: str, when) -> float:
        """Close on the first trading day ≥ `when` (the publication date). NaN if unknown or
        if the publication post-dates the panel — never the last known price, which would be
        a look-ahead for pre-panel periods and a stale price for post-panel ones."""
        entry = self._price_lookup().get(ticker)
        if entry is None:
            return float("nan")
        dates, vals = entry
        t = np.datetime64(pd.Timestamp(when).to_datetime64())
        i = int(np.searchsorted(dates, t, side="left"))
        if i >= len(dates):
            return float("nan")
        # tolerate weekends/holidays but not a multi-week gap (delisting / pre-IPO)
        if (dates[i] - t).astype("timedelta64[D]").astype(int) > 7:
            return float("nan")
        return float(vals[i])

    # ---------------------------------------------------------------- extraction
    def _concept_facts(self, facts: dict, tags: list[str]) -> list[dict]:
        """Every usable fact across the tag list, stamped with a preference `rank`.

        Filers migrate tags mid-history (NVDA reported revenue as
        `RevenueFromContractWithCustomerExcludingAssessedTax` until 2020, then `Revenues`),
        so taking only the first tag that has *any* facts silently truncates the series.
        We pool them and let `_dedupe_earliest_filed` prefer the lower-ranked tag wherever
        two tags cover the same period.
        """
        rows = []
        for rank, tag in enumerate(tags):
            for taxonomy in ("us-gaap", "dei", "ifrs-full"):
                block = facts.get(taxonomy, {})
                if tag not in block:
                    continue
                for unit, items in block[tag]["units"].items():
                    for it in items:
                        if it.get("form") not in ACCEPTED_FORMS or it.get("val") is None:
                            continue
                        rows.append({**it, "unit": unit, "tag": tag, "rank": rank})
                break               # same tag under another taxonomy would just duplicate
        return rows

    def ledger(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        """Point-in-time fundamentals ledger for one ticker: one row per fiscal quarter with
        TTM flows, period-end stocks, the publication date, and a PIT price/market cap.

        Columns: period_end, publication_date, fiscal_period, reported_currency, revenue,
        eps, ebitda, free_cash_flow, book_value, shares, price, market_cap, enterprise_value.
        """
        cache = Path(self.cache_dir) / f"{ticker}.json"
        if cache.exists() and not refresh:
            df = pd.DataFrame(json.loads(cache.read_text()))
        else:
            df = self._build_ledger(ticker)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(df.to_json(orient="records"))
        if df.empty:
            return df
        # Put as-reported per-share quantities on the price panel's (today's) split basis.
        # Done outside the cache so the cached XBRL extraction stays raw and re-adjustable.
        sp = (self.splits or {}).get(ticker)
        if sp:
            fac = np.array([split_factor(sp, pe) for pe in df["period_end"]], dtype="float64")
            df["shares"] = df["shares"] * fac
            df["eps"] = df["eps"] / fac
        # Prices are cheap and depend on the panel, so they are attached after the cache.
        df["price"] = [self.price_on(ticker, d) for d in df["publication_date"]]
        df["market_cap"] = df["price"] * df["shares"]
        df["enterprise_value"] = [
            enterprise_value(mc, lt, st, csh)
            for mc, lt, st, csh in zip(df["market_cap"], df["debt_lt"], df["debt_st"], df["cash"])
        ]
        return df

    def _build_ledger(self, ticker: str) -> pd.DataFrame:
        cands = list(self.cik_candidates().get(ticker.upper(), []))
        if ticker.upper() in TICKER_CIK_OVERRIDES:
            cands.insert(0, {"cik": TICKER_CIK_OVERRIDES[ticker.upper()],
                             "name": ticker.upper(), "exchange": None})
        if not cands:
            logger.warning("no CIK for %s", ticker)
            return pd.DataFrame()
        facts: dict = {}
        for info in cands:                      # first CIK with real us-gaap facts wins
            try:
                payload = self._get_json(SEC_FACTS.format(cik=info["cik"]))
            except urllib.error.HTTPError as e:
                logger.warning("companyfacts %s (CIK %s) -> HTTP %s", ticker, info["cik"], e.code)
                continue
            f = payload.get("facts", {})
            if f.get("us-gaap") or f.get("ifrs-full"):
                facts = f
                break
        if not facts:
            return pd.DataFrame()

        frames, units = {}, {}
        # A real fiscal quarter end is one a *statement* line item reports on. Cover-page
        # facts (dei share counts, dated near the filing) would otherwise inject phantom
        # periods a few weeks after each quarter, inflating the row count and diluting every
        # coverage percentage in the audit.
        fiscal_ends: set[str] = set()
        for name, (kind, tags) in CONCEPTS.items():
            raw = self._concept_facts(facts, tags)
            if not raw:
                continue
            units[name] = raw[0]["unit"]
            if kind == "duration":
                q = quarterly_from_duration_facts(raw)
                if not q:
                    continue
                fiscal_ends.update(r["end"] for r in q)
                s = pd.DataFrame(q).set_index("end")
                frames[name] = pd.DataFrame({
                    name: ttm(s), f"{name}__filed": s["filed"]})
            elif kind == "level":
                # Only DIRECT quarterly facts: differencing two year-to-date share counts
                # would produce nonsense, since a share count is a level, not a flow.
                q = [r for r in quarterly_from_duration_facts(raw) if r["source"] == "direct"]
                if not q:
                    continue
                s = pd.DataFrame(q).set_index("end")
                frames[name] = pd.DataFrame({name: s["val"], f"{name}__filed": s["filed"]})
            else:
                inst = _dedupe_earliest_filed(
                    [f for f in raw if not f.get("start")] or raw, key=lambda f: f["end"])
                s = pd.DataFrame(inst).set_index("end")
                if name == "book_value":
                    fiscal_ends.update(s.index)
                frames[name] = pd.DataFrame({name: s["val"], f"{name}__filed": s["filed"]})

        if not frames:
            return pd.DataFrame()

        led = pd.concat(frames.values(), axis=1).sort_index()
        led.index.name = "period_end"
        led = led.reset_index()

        filed_cols = [c for c in led.columns if c.endswith("__filed")]
        # Publication date = the LATEST filed among the fields on that row: the first moment
        # every input on this row was public. Conservative by construction.
        led["publication_date"] = led[filed_cols].max(axis=1)
        led = led.drop(columns=filed_cols)
        led = led.dropna(subset=["publication_date"])

        for c in FLOW_COLS + STOCK_COLS + ["shares_wavg"]:
            if c not in led.columns:
                led[c] = np.nan
        led["shares"] = led["shares"].fillna(led["shares_wavg"])

        # Balance-sheet items are often tagged only in the 10-K (share count especially).
        # Carrying the last PUBLISHED value forward (≤4 quarters) uses older information,
        # never future information, so it cannot leak — it can only be stale.
        led[STOCK_COLS] = led[STOCK_COLS].ffill(limit=4)

        led["ebitda"] = led["operating_income"] + led["dep_amort"]
        led["free_cash_flow"] = led["cfo"] - led["capex"]
        led["ticker"] = ticker
        led["reported_currency"] = units.get("revenue") or units.get("book_value") or "USD"
        led["fiscal_period"] = "TTM"
        keep = ["ticker", "period_end", "publication_date", "fiscal_period", "reported_currency",
                "revenue", "eps", "ebitda", "free_cash_flow", "book_value", "cash",
                "debt_lt", "debt_st", "shares"]
        led = led[keep]
        led = led[led["period_end"].isin(fiscal_ends)]          # drop cover-page phantoms
        # Only quarters where at least one value input survived are worth carrying.
        led = led.dropna(subset=["revenue", "eps", "book_value", "ebitda", "free_cash_flow"],
                         how="all")
        return led.reset_index(drop=True)

    # ---------------------------------------------------------------- FMP-shaped API
    @staticmethod
    def fetch_splits(tickers, start="2015-01-01", cache_dir: Path = SEC_CACHE,
                     batch: int = 100, refresh: bool = False) -> dict:
        """{ticker: [(date, ratio), …]} split history from yfinance, batched and cached.

        SEC XBRL has no reliable split tag, so the corporate-action history comes from the
        same vendor that adjusted the price panel — which is exactly the basis we need to
        match. Returns {} (and logs) if yfinance is unavailable, in which case per-share
        quantities stay as-reported and the audit reports the adjustment as not applied.
        """
        cf = Path(cache_dir) / "splits.json"
        if cf.exists() and not refresh:
            return json.loads(cf.read_text())
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance unavailable — split adjustment NOT applied")
            return {}
        tickers = sorted(set(tickers))
        out: dict[str, list] = {}
        for i in range(0, len(tickers), batch):
            chunk = tickers[i:i + batch]
            try:
                d = yf.download(chunk, start=start, actions=True, progress=False,
                                auto_adjust=False, group_by="column", threads=True)
            except Exception as e:                   # noqa: BLE001 — best effort
                logger.warning("split download failed for chunk %d: %s", i // batch, e)
                continue
            if "Stock Splits" not in d.columns.get_level_values(0):
                continue
            sp = d["Stock Splits"]
            if isinstance(sp, pd.Series):            # single-ticker shape
                sp = sp.to_frame(chunk[0])
            for tk in sp.columns:
                ev = sp[tk][sp[tk].fillna(0) != 0]
                if len(ev):
                    out[str(tk)] = [[str(pd.Timestamp(d0).date()), float(v)]
                                    for d0, v in ev.items()]
            logger.info("splits: %d/%d tickers scanned", min(i + batch, len(tickers)), len(tickers))
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(out))
        return out

    def statements(self, ticker: str, quarters: int = 12) -> list[dict]:
        """Newest-first records matching `FMPClient.statements`, so `run_audit` is source-agnostic."""
        led = self.ledger(ticker)
        if led.empty:
            return []
        led = led.sort_values("period_end", ascending=False).head(quarters)
        out = []
        for r in led.to_dict("records"):
            out.append({
                "ticker": ticker,
                "period_end": r["period_end"],
                "publication_date": r["publication_date"],
                "period": r["fiscal_period"],
                "reported_currency": r["reported_currency"],
                "revenue": r["revenue"],
                "eps": r["eps"],
                "ebitda": r["ebitda"],
                "book_value": r["book_value"],
                "free_cash_flow": r["free_cash_flow"],
                "enterprise_value": r["enterprise_value"],
                "market_cap": r["market_cap"],
                "price": r["price"],
            })
        return out
