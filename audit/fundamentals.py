"""
Fundamentals data-quality audit — Phase 17. The GO/NO-GO gate before the value factor.

⚠️ EDUCATIONAL SIMULATION context. This module VERIFIES data; it never invents it. If no
FMP key / no network is available, the audit cannot run and the harness says so — it does
NOT emit a fake report.

The seven checks (per the #17 spec)
-----------------------------------
1. Accuracy   — spot-check ~10 (ticker, period) values vs a second source (yfinance);
                report the max relative discrepancy per field.
2. Coverage   — % of the universe carrying each value-factor input; gaps by sector/size.
3. Point-in-time — every fundamental must carry a real publication date (FMP `fillingDate`)
                and `fillingDate > period_end`; the value factor is built off the LAGGED
                publication date, never period-end. This is the leakage gate.
4. Outliers   — flag impossible values (negative equity, zero denominators, extreme ratios)
                and document handling (winsorize).
5. Consistency— units, USD currency, fiscal-year alignment, per-share split adjustment.
6. Survivorship — does history include delisted names or only survivors? Documented.
7. Reproducible — this script + a written report; re-runnable, disk-cached, no eyeballing.

Data sources
------------
* **`sec` (default)** — SEC EDGAR XBRL `companyfacts`. Every fact carries `filed`, the real
  EDGAR publication date, so the point-in-time gate runs against the primary source. Free,
  keyless, reachable. See `audit/sec_provider.py`.
* **`fmp`** — Financial Modeling Prep free tier (the source #17 was originally specified
  against). Same record shape via `FMPClient`. Retained, but unreachable from this machine.

yfinance is a cross-check source only, for Check 1 — it carries no publication dates, so it
can never be the point-in-time source.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fund_audit")

FMP_BASE = "https://financialmodelingprep.com/stable"   # v3 legacy retired 2025-08-31
CACHE_DIR = Path("data/fundamentals_cache")   # gitignored; raw FMP JSON, so re-runs are free
REPORT_PATH = Path("figures/audit/fundamentals_audit.md")

# The raw fields each value-factor input (from #18) is derived from.
#   earnings yield  = eps / price          book-to-market = book_value / market_cap
#   EBITDA/EV       = ebitda / ev          FCF yield      = fcf / market_cap
CORE_INPUTS = ["eps", "price", "book_value", "ebitda", "enterprise_value",
               "free_cash_flow", "market_cap"]

# ~10 hand-verifiable (ticker, fiscal-period) accuracy spot-checks. `period` is the FMP
# statement `date` (period end); we compare FMP vs yfinance for these fields.
SPOT_CHECKS = [
    ("AAPL", "revenue"), ("AAPL", "eps"), ("AAPL", "book_value"),
    ("MSFT", "revenue"), ("MSFT", "eps"), ("MSFT", "book_value"),
    ("NVDA", "revenue"), ("JPM", "book_value"), ("XOM", "revenue"), ("KO", "eps"),
]

# GO/NO-GO thresholds — conservative, documented, not tuned.
TH_ACCURACY_MAX_REL = 0.05    # ≤5% max relative discrepancy vs cross-source per field
TH_COVERAGE_HARD = 0.70       # any core input below this ⇒ NO-GO
TH_COVERAGE_SOFT = 0.90       # below this ⇒ flag (not a hard fail)
WINSOR_PCT = (0.01, 0.99)     # ratio winsorization bounds used downstream (#18)


# ============================================================ pure logic (unit-tested)
def winsorize(x, pct=WINSOR_PCT) -> np.ndarray:
    """Clip to the [lo, hi] cross-sectional percentiles so outliers can't swing ranks."""
    a = np.asarray(x, dtype="float64")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return a
    lo, hi = np.quantile(finite, pct[0]), np.quantile(finite, pct[1])
    return np.clip(a, lo, hi)


def zscore(x) -> np.ndarray:
    """Cross-sectional z-score, NaN-safe (population std). All-equal ⇒ zeros."""
    a = np.asarray(x, dtype="float64")
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0:
        return np.where(np.isfinite(a), 0.0, np.nan)
    return (a - mu) / sd


def value_ratios(rec: dict) -> dict:
    """The four value inputs from #18, oriented so higher = cheaper = better. Missing or
    undefined (zero/negative denominator) ⇒ NaN, so the audit can measure real coverage."""
    def safe_div(n, d, allow_neg_num=True):
        if n is None or d is None:
            return float("nan")
        try:
            n, d = float(n), float(d)
        except (TypeError, ValueError):
            return float("nan")
        if d <= 0 or not math.isfinite(n) or not math.isfinite(d):
            return float("nan")
        if not allow_neg_num and n < 0:
            return float("nan")
        return n / d
    return {
        "earnings_yield": safe_div(rec.get("eps"), rec.get("price")),
        "book_to_market": safe_div(rec.get("book_value"), rec.get("market_cap"),
                                   allow_neg_num=False),  # negative equity ⇒ undefined
        "ebitda_to_ev": safe_div(rec.get("ebitda"), rec.get("enterprise_value")),
        "fcf_yield": safe_div(rec.get("free_cash_flow"), rec.get("market_cap")),
    }


def detect_outliers(rec: dict) -> list[str]:
    """Return a list of outlier/impossibility flags for one record (Check 4)."""
    flags = []
    bv = rec.get("book_value")
    if bv is not None and _num(bv) is not None and _num(bv) < 0:
        flags.append("negative_equity")           # P/B undefined
    for k in ("price", "market_cap", "enterprise_value"):
        v = _num(rec.get(k))
        if v is not None and v <= 0:
            flags.append(f"nonpositive_{k}")
    ratios = value_ratios(rec)
    ey = ratios.get("earnings_yield")
    if ey is not None and math.isfinite(ey) and abs(ey) > 2.0:   # |E/P|>200% ⇒ implausible
        flags.append("extreme_earnings_yield")
    # A book-to-market above 10 means the market values the firm at under a tenth of book —
    # for a large cap that is nearly always a broken share count (dual-class filers tag only
    # class-A-equivalent shares), not a real valuation. Catch it here rather than in the book.
    bm = ratios.get("book_to_market")
    if bm is not None and math.isfinite(bm) and bm > 10.0:
        flags.append("extreme_book_to_market")
    return flags


def assert_point_in_time(records: list[dict]) -> dict:
    """Check 3 — the leakage gate. Every record must carry a real publication date and it
    must fall strictly AFTER the period end. Returns violation counts + offending samples.

    A record is {ticker, period_end, publication_date (fillingDate), ...}. Missing a
    publication date, or publication_date <= period_end, is a leakage risk.
    """
    missing_date, not_after, ok = [], [], 0
    for r in records:
        pe, pub = r.get("period_end"), r.get("publication_date")
        if not pub:
            missing_date.append(r)
            continue
        pe_t, pub_t = pd.to_datetime(pe, errors="coerce"), pd.to_datetime(pub, errors="coerce")
        if pd.isna(pub_t):
            missing_date.append(r)
        elif pd.isna(pe_t) or pub_t <= pe_t:
            not_after.append(r)
        else:
            ok += 1
    n = len(records)
    return {
        "n_records": n,
        "n_ok": ok,
        "n_missing_publication_date": len(missing_date),
        "n_publication_not_after_period": len(not_after),
        "pass": n > 0 and not missing_date and not not_after,
        "sample_missing": [_sample(r) for r in missing_date[:5]],
        "sample_not_after": [_sample(r) for r in not_after[:5]],
    }


def coverage_map(records: list[dict], fields=CORE_INPUTS) -> dict:
    """Check 2 — fraction of records with a usable (non-null, finite) value per field."""
    if not records:
        return {f: 0.0 for f in fields}
    out = {}
    for f in fields:
        present = sum(1 for r in records if _num(r.get(f)) is not None)
        out[f] = present / len(records)
    return out


def _num(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sample(r: dict) -> dict:
    return {k: r.get(k) for k in ("ticker", "period_end", "publication_date")}


# ============================================================ FMP client (network path)
@dataclass
class FMPClient:
    """Thin FMP free-tier client with on-disk JSON caching + polite rate-limit backoff.
    Never fabricates: a failed fetch raises / returns [] and is surfaced in the report."""
    api_key: str
    cache_dir: Path = CACHE_DIR
    min_interval: float = 0.30          # seconds between calls (free-tier friendly)
    _last: float = field(default=0.0, repr=False)

    def _get(self, path: str, **params) -> list | dict:
        params["apikey"] = self.api_key
        qs = urllib.parse.urlencode(params)
        cache_key = urllib.parse.quote(f"{path}?{qs}".replace("apikey=" + self.api_key,
                                                              "apikey=KEY"), safe="")
        cf = self.cache_dir / f"{cache_key}.json"
        if cf.exists():
            return json.loads(cf.read_text())
        # polite pacing
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        url = f"{FMP_BASE}/{path}?{qs}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.loads(resp.read().decode())
                self._last = time.monotonic()
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cf.write_text(json.dumps(data))
                return data
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 3:      # rate limited — back off
                    time.sleep(2 ** attempt)
                    continue
                raise
        return []

    def statements(self, ticker: str, quarters: int = 12) -> list[dict]:
        """Merge income / balance-sheet / cash-flow / enterprise-value / profile into tidy
        per-(ticker, period_end, publication_date) records with the CORE_INPUTS fields.

        Uses FMP's current `stable` API: endpoints take `?symbol=`, and the publication
        date field is `filingDate` (the legacy v3 spelling was `fillingDate`)."""
        inc = self._get("income-statement", symbol=ticker, period="quarter", limit=quarters)
        bal = self._get("balance-sheet-statement", symbol=ticker, period="quarter", limit=quarters)
        cfs = self._get("cash-flow-statement", symbol=ticker, period="quarter", limit=quarters)
        ev = self._get("enterprise-values", symbol=ticker, period="quarter", limit=quarters)
        prof = self._get("profile", symbol=ticker)
        prof0 = prof[0] if isinstance(prof, list) and prof else {}

        bal_by = {r.get("date"): r for r in bal} if isinstance(bal, list) else {}
        cfs_by = {r.get("date"): r for r in cfs} if isinstance(cfs, list) else {}
        ev_by = {r.get("date"): r for r in ev} if isinstance(ev, list) else {}

        out = []
        for r in (inc if isinstance(inc, list) else []):
            d = r.get("date")
            b, c, e = bal_by.get(d, {}), cfs_by.get(d, {}), ev_by.get(d, {})
            out.append({
                "ticker": ticker,
                "period_end": d,
                # publication date: `filingDate` (stable) → `fillingDate` (legacy) → accepted
                "publication_date": (r.get("filingDate") or r.get("fillingDate")
                                     or r.get("acceptedDate")),
                "period": r.get("period"),
                "reported_currency": r.get("reportedCurrency"),
                "revenue": r.get("revenue"),
                "eps": r.get("epsDiluted") if r.get("epsDiluted") is not None else r.get("eps"),
                "ebitda": r.get("ebitda"),
                "book_value": b.get("totalStockholdersEquity"),
                "free_cash_flow": c.get("freeCashFlow"),
                "enterprise_value": e.get("enterpriseValue"),
                "market_cap": e.get("marketCapitalization") or prof0.get("marketCap"),
                "price": prof0.get("price") or e.get("stockPrice"),
            })
        return out


def _load_key(explicit=None) -> str | None:
    """Key precedence: --fmp-key > FMP_API_KEY env > .env file. Never logged."""
    import os
    if explicit:
        return explicit.strip()
    if os.environ.get("FMP_API_KEY"):
        return os.environ["FMP_API_KEY"].strip()
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.strip().startswith("FMP_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ============================================================ accuracy cross-check (Check 1)
def accuracy_vs_yfinance(records_by_ticker: dict, spot_checks=SPOT_CHECKS) -> dict:
    """Compare our newest value against yfinance for the spot-check fields.

    Like-for-like matters: our flow fields (revenue, eps) are TRAILING-TWELVE-MONTH, so the
    cross-source value must be TTM too — yfinance's four most recent quarterly columns
    summed, and its own `trailingEps`. `book_value` is a balance-sheet level, compared at
    the latest quarter.

    yfinance is flaky (rate limits) and has no publication dates, so a failure here is
    reported as 'cross-check unavailable', NOT as an audit failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"status": "yfinance_not_installed", "checks": []}
    results = []
    for ticker, field_name in spot_checks:
        recs = records_by_ticker.get(ticker) or []
        if not recs:
            results.append({"ticker": ticker, "field": field_name, "status": "no_source_data"})
            continue
        ours = _num(recs[0].get(field_name))
        try:
            yv = _yf_value(yf, ticker, field_name)
        except Exception as e:                       # noqa: BLE001 — yfinance is best-effort
            results.append({"ticker": ticker, "field": field_name,
                            "status": f"xcheck_unavailable:{type(e).__name__}"})
            continue
        if ours is None or yv is None or yv == 0:
            results.append({"ticker": ticker, "field": field_name, "status": "incomparable",
                            "ours": ours, "yf": yv})
            continue
        rel = abs(ours - yv) / abs(yv)
        results.append({"ticker": ticker, "field": field_name, "status": "ok",
                        "ours": ours, "yf": yv, "rel_discrepancy": rel,
                        "period_end": recs[0].get("period_end")})
    comparable = [r["rel_discrepancy"] for r in results if r.get("status") == "ok"]
    return {
        "status": "ran" if comparable else "no_comparisons",
        "max_rel_discrepancy": (max(comparable) if comparable else None),
        "median_rel_discrepancy": (float(np.median(comparable)) if comparable else None),
        "n_comparable": len(comparable),
        "checks": results,
    }


def _yf_value(yf, ticker: str, field_name: str):
    """TTM-matched cross-source value, or None if yfinance doesn't carry it."""
    t = yf.Ticker(ticker)
    if field_name == "revenue":
        df = t.quarterly_financials
        for k in ("Total Revenue", "Operating Revenue"):
            if k in df.index:
                q = pd.to_numeric(df.loc[k], errors="coerce").dropna()
                return float(q.iloc[:4].sum()) if len(q) >= 4 else None
        return None
    if field_name == "eps":
        return t.info.get("trailingEps")             # already TTM
    if field_name == "book_value":
        df = t.quarterly_balance_sheet
        for k in ("Stockholders Equity", "Total Stockholder Equity"):
            if k in df.index:
                q = pd.to_numeric(df.loc[k], errors="coerce").dropna()
                return float(q.iloc[0]) if len(q) else None
        return None
    return None


# ============================================================ orchestration + report
def make_provider(source: str = "sec", api_key: str | None = None,
                  cache_dir: Path | None = None, price_panel=None):
    """Build the fundamentals provider. Both expose `.statements(ticker, quarters)` returning
    identically-shaped records, so every check below is source-agnostic."""
    if source == "sec":
        from audit.sec_provider import SECClient, SEC_CACHE
        return SECClient(cache_dir=Path(cache_dir or SEC_CACHE), price_panel=price_panel)
    if source == "fmp":
        if not api_key:
            raise ValueError("source='fmp' needs an API key")
        return FMPClient(api_key=api_key, cache_dir=Path(cache_dir or CACHE_DIR))
    raise ValueError(f"unknown source {source!r} (expected 'sec' or 'fmp')")


def run_audit(tickers: list[str], api_key: str | None = None, quarters: int = 12,
              cache_dir: Path | None = None, source: str = "sec",
              price_panel=None, provider=None) -> dict:
    """Fetch fundamentals for `tickers` and run all seven checks.

    Requires live connectivity to the chosen source — raises if the very first fetch fails,
    so the audit can never 'pass' on no data. `price_panel` (the daily OHLCV panel) lets the
    SEC provider strike a point-in-time price/market cap at each publication date.
    """
    client = provider or make_provider(source, api_key, cache_dir, price_panel)
    # Per-share facts are as-reported; the price panel is on today's split basis. Line them
    # up before anything is computed, or every pre-split name looks spuriously cheap.
    if getattr(client, "splits", "n/a") is None:
        client.splits = client.fetch_splits(tickers)
    by_ticker, all_records, fetch_errors = {}, [], []
    for i, tk in enumerate(tickers):
        try:
            recs = client.statements(tk, quarters=quarters)
        except Exception as e:                       # noqa: BLE001
            fetch_errors.append({"ticker": tk, "error": f"{type(e).__name__}: {e}"})
            if i == 0:
                raise RuntimeError(
                    f"FMP fetch failed on the first ticker ({tk}): {e}. Aborting rather "
                    f"than auditing empty data.") from e
            continue
        by_ticker[tk] = recs
        all_records.extend(recs)
        if (i + 1) % 25 == 0:
            logger.info("fetched %d/%d tickers", i + 1, len(tickers))

    latest = [recs[0] for recs in by_ticker.values() if recs]   # newest period per name

    checks = {
        "accuracy": accuracy_vs_yfinance(by_ticker),
        "coverage": coverage_map(latest),
        "point_in_time": assert_point_in_time(all_records),
        "outliers": _outlier_summary(latest),
        "consistency": _consistency_summary(latest, splits=getattr(client, "splits", None)),
        "survivorship": {
            "universe_is_point_in_time": False,
            "note": "Universe is today's members (survivorship-biased). History omits "
                    "delisted names; a point-in-time source (e.g. Sharadar) is required for "
                    "unbiased backtests. Results are DIRECTIONAL / educational only. Note "
                    "the FUNDAMENTALS themselves are point-in-time (EDGAR `filed` dates, "
                    "earliest-filed value per period, no restatements) — the survivorship "
                    "caveat is about which TICKERS are in the list, not about the data.",
        },
    }
    report = {
        "source": source,
        "n_tickers_requested": len(tickers),
        "n_tickers_with_data": len(by_ticker),
        "n_records": len(all_records),
        "quarters_per_ticker": quarters,
        "fetch_errors": fetch_errors,
        "checks": checks,
    }
    report["go_no_go"] = go_no_go(report)
    return report


def _outlier_summary(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    flagged = []
    for r in records:
        fl = detect_outliers(r)
        for f in fl:
            counts[f] = counts.get(f, 0) + 1
        if fl:
            flagged.append({"ticker": r.get("ticker"), "flags": fl})
    return {"counts": counts, "n_flagged": len(flagged), "sample": flagged[:10],
            "handling": f"winsorize ratios to {WINSOR_PCT} pct; drop undefined (neg equity / "
                        f"nonpositive denominator) before z-scoring (#18)."}


def _consistency_summary(records: list[dict], splits: dict | None = None) -> dict:
    currencies = sorted({r.get("reported_currency") for r in records if r.get("reported_currency")})
    periods = sorted({r.get("period") for r in records if r.get("period")})
    audited = {r.get("ticker") for r in records}
    split_names = sorted(audited & set(splits or {}))
    return {
        "currencies_seen": currencies,
        "all_usd": currencies == ["USD"] or currencies == [],
        "periods_seen": periods,
        "split_adjustment_applied": splits is not None,
        "n_names_with_splits": len(split_names),
        "split_names_sample": split_names[:12],
        "note": "Absolute-$ line items are as reported (XBRL facts are already in whole "
                "units, not thousands). SPLITS: XBRL facts are as-reported and never "
                "retro-adjusted (we keep the earliest-filed value on purpose), while the "
                "yfinance price panel rewrites history onto today's split basis — so EPS "
                "and share counts are multiplied by the cumulative post-period split factor "
                "before any ratio is formed. Without that step NVDA's pre-June-2024 "
                "earnings yield would read 10x too high. Fiscal calendars differ across "
                "filers by design; every record is keyed to its own fiscal period end and "
                "joined by publication date, never by calendar date. Non-USD filers are "
                "flagged above.",
    }


def go_no_go(report: dict) -> dict:
    """Decide GO / NO-GO from the checks, with explicit reasons. NO-GO is triggered by
    leakage risk (point-in-time), thin coverage, or large accuracy discrepancies."""
    reasons, verdict = [], "GO"
    c = report["checks"]

    pit = c["point_in_time"]
    if not pit.get("pass"):
        verdict = "NO-GO"
        reasons.append(
            f"Point-in-time FAIL: {pit.get('n_missing_publication_date',0)} records missing a "
            f"publication date, {pit.get('n_publication_not_after_period',0)} with "
            f"publication ≤ period-end (leakage risk).")

    cov = c["coverage"]
    low = {k: v for k, v in cov.items() if v < TH_COVERAGE_HARD}
    soft = {k: v for k, v in cov.items() if TH_COVERAGE_HARD <= v < TH_COVERAGE_SOFT}
    if low:
        verdict = "NO-GO"
        reasons.append("Coverage FAIL (<70%): "
                       + ", ".join(f"{k} {v:.0%}" for k, v in low.items()))
    if soft:
        reasons.append("Coverage WARN (70–90%): "
                       + ", ".join(f"{k} {v:.0%}" for k, v in soft.items()))

    acc = c["accuracy"]
    if acc.get("status") == "ran" and acc.get("max_rel_discrepancy") is not None:
        if acc["max_rel_discrepancy"] > TH_ACCURACY_MAX_REL:
            verdict = "NO-GO"
            reasons.append(f"Accuracy FAIL: max discrepancy vs cross-source "
                           f"{acc['max_rel_discrepancy']:.1%} > {TH_ACCURACY_MAX_REL:.0%}.")
    else:
        reasons.append("Accuracy cross-check unavailable (yfinance) — spot-check manually "
                       "before trusting; not a hard fail on its own.")

    reasons.append("Survivorship: universe is NOT point-in-time — GO covers DIRECTIONAL / "
                   "educational use only, never live/point-in-time claims.")
    return {"verdict": verdict, "reasons": reasons}


def write_report(report: dict, path: Path = REPORT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = report["checks"]
    g = report["go_no_go"]
    L = []
    src = report.get("source", "sec")
    src_label = {"sec": "SEC EDGAR XBRL `companyfacts` (primary source, real `filed` dates)",
                 "fmp": "Financial Modeling Prep free tier"}.get(src, src)
    L.append("# RankAlpha — fundamentals data-quality audit (Phase 17)\n")
    L.append(f"*Reproducible via `python scripts/audit_fundamentals.py --source {src}`. "
             f"Educational SIMULATION. Source: {src_label}; yfinance cross-check only.*\n")
    L.append(f"## VERDICT: **{g['verdict']}**\n")
    for r in g["reasons"]:
        L.append(f"- {r}")
    L.append("")
    L.append(f"Universe: {report['n_tickers_with_data']}/{report['n_tickers_requested']} "
             f"tickers returned data · {report['n_records']} statement records "
             f"({report['quarters_per_ticker']} quarters/name).\n")

    L.append("## 1. Accuracy (vs yfinance, TTM-matched)")
    acc = c["accuracy"]
    L.append(f"- status: `{acc['status']}` · comparable fields: {acc.get('n_comparable',0)} · "
             f"max relative discrepancy: "
             f"{'%.2f%%' % (acc['max_rel_discrepancy']*100) if acc.get('max_rel_discrepancy') is not None else 'n/a'}"
             f" · median: "
             f"{'%.2f%%' % (acc['median_rel_discrepancy']*100) if acc.get('median_rel_discrepancy') is not None else 'n/a'}")
    if acc.get("checks"):
        L.append("")
        L.append("| ticker | field | ours | yfinance | rel. diff | status |\n|---|---|---|---|---|---|")
        for r in acc["checks"]:
            o, y = r.get("ours"), r.get("yf")
            rel = r.get("rel_discrepancy")
            L.append(f"| {r['ticker']} | {r['field']} | "
                     f"{'' if o is None else f'{o:,.4g}'} | {'' if y is None else f'{y:,.4g}'} | "
                     f"{'' if rel is None else f'{rel:.2%}'} | `{r['status']}` |")
    L.append("")

    L.append("## 2. Coverage (% of universe with each value input)")
    L.append("| field | coverage |\n|---|---|")
    for k, v in c["coverage"].items():
        L.append(f"| {k} | {v:.1%} |")
    L.append("")

    L.append("## 3. Point-in-time integrity (leakage gate)")
    pit = c["point_in_time"]
    L.append(f"- records: {pit['n_records']} · OK: {pit['n_ok']} · "
             f"missing publication date: {pit['n_missing_publication_date']} · "
             f"publication ≤ period-end: {pit['n_publication_not_after_period']} · "
             f"**pass: {pit['pass']}**")
    L.append("")

    L.append("## 4. Outliers")
    L.append(f"- counts: `{c['outliers']['counts']}` · flagged names: {c['outliers']['n_flagged']}")
    L.append(f"- handling: {c['outliers']['handling']}")
    L.append("")

    L.append("## 5. Consistency")
    con = c["consistency"]
    L.append(f"- currencies: `{con['currencies_seen']}` · all-USD: {con['all_usd']} · "
             f"periods: `{con['periods_seen']}`")
    L.append(f"- split adjustment applied: **{con.get('split_adjustment_applied')}** · "
             f"names with splits in window: {con.get('n_names_with_splits')} "
             f"(e.g. {', '.join(con.get('split_names_sample', [])) or 'none'})")
    L.append(f"- {con['note']}")
    L.append("")

    L.append("## 6. Survivorship")
    L.append(f"- {c['survivorship']['note']}")
    L.append("")

    if report.get("fetch_errors"):
        L.append(f"## Fetch errors ({len(report['fetch_errors'])})")
        for e in report["fetch_errors"][:20]:
            L.append(f"- {e['ticker']}: {e['error']}")
        L.append("")

    path.write_text("\n".join(L))
    logger.info("wrote report -> %s", path)
    return path


# ============================================================ offline self-test
def self_test() -> dict:
    """Exercise the PURE logic on synthetic fixtures — proves the checks work without any
    network. This is NOT an audit of real data; it validates the harness itself."""
    results = {}

    # winsorize clips the tails
    w = winsorize([1, 2, 3, 4, 1000], pct=(0.0, 0.8))
    results["winsorize_clips"] = float(w.max()) < 1000

    # z-score is mean-0
    z = zscore([1.0, 2.0, 3.0, 4.0])
    results["zscore_mean0"] = abs(float(np.nanmean(z))) < 1e-9

    # value ratios: negative equity ⇒ book_to_market NaN; good record ⇒ finite
    good = {"eps": 6.0, "price": 100.0, "book_value": 50e9, "market_cap": 200e9,
            "ebitda": 30e9, "enterprise_value": 210e9, "free_cash_flow": 20e9}
    neg = {**good, "book_value": -1e9}
    results["ratio_good_finite"] = math.isfinite(value_ratios(good)["book_to_market"])
    results["ratio_negequity_nan"] = math.isnan(value_ratios(neg)["book_to_market"])

    # outliers flag negative equity + nonpositive price
    results["outlier_flags"] = set(detect_outliers({"book_value": -1, "price": 0})) >= {
        "negative_equity", "nonpositive_price"}

    # point-in-time: leaking record (pub <= period end) and missing-date record are caught
    recs = [
        {"ticker": "A", "period_end": "2024-03-31", "publication_date": "2024-05-01"},  # ok
        {"ticker": "B", "period_end": "2024-03-31", "publication_date": "2024-03-31"},  # leak
        {"ticker": "C", "period_end": "2024-03-31", "publication_date": None},          # missing
    ]
    pit = assert_point_in_time(recs)
    results["pit_catches_leak"] = (pit["n_ok"] == 1 and pit["n_publication_not_after_period"] == 1
                                   and pit["n_missing_publication_date"] == 1 and not pit["pass"])

    # coverage: field present in 2/3 records ⇒ ~0.667
    cov = coverage_map([{"eps": 1}, {"eps": None}, {"eps": 3}], fields=["eps"])
    results["coverage_fraction"] = abs(cov["eps"] - 2/3) < 1e-9

    # go/no-go: a leaking + thin-coverage report must be NO-GO
    fake = {
        "n_tickers_requested": 3, "n_tickers_with_data": 3, "n_records": 3,
        "quarters_per_ticker": 1, "fetch_errors": [],
        "checks": {
            "accuracy": {"status": "no_comparisons", "max_rel_discrepancy": None, "n_comparable": 0, "checks": []},
            "coverage": {k: 0.5 for k in CORE_INPUTS},
            "point_in_time": pit,
            "outliers": _outlier_summary([neg]),
            "consistency": _consistency_summary([good]),
            "survivorship": {"note": "x"},
        },
    }
    results["gonogo_blocks_bad_data"] = go_no_go(fake)["verdict"] == "NO-GO"

    results["all_passed"] = all(results.values())
    return results


if __name__ == "__main__":
    from pprint import pprint
    pprint(self_test())
