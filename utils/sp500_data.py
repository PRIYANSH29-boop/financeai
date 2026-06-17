"""
S&P 500 panel data builder — RankAlpha Phase 1 (DATA).

Scrapes the current S&P 500 constituent list from Wikipedia, downloads ~7 years of
daily OHLCV from yfinance in gentle batches, and stores a tidy *panel* parquet:

    date, ticker, open, high, low, close, adj_close, volume

Use `adj_close` for returns. This is Phase 1 only — NO features, NO labels here.

⚠️ SURVIVORSHIP BIAS: v1 uses the *current* S&P 500 membership for the whole history.
Names that were dropped/delisted over the last 7 years are silently excluded, which
biases any backtest upward. The fix (point-in-time constituents) is deferred — see
README and ROADMAP.md.
"""

import io
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sp500_data")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "RankAlpha/0.1 (educational quant research; contact priyansh2005p@gmail.com)"

# Tidy panel schema — order matters for the saved parquet.
PANEL_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


class SP500DataBuilder:
    """Builds and stores the S&P 500 daily price panel."""

    def __init__(self, data_dir="data", years=7, batch_size=50):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.years = years
        self.batch_size = batch_size
        # End = today; start = `years` ago. yfinance `end` is exclusive, so push +1 day.
        self.end = datetime.now().date()
        self.start = self.end - timedelta(days=int(round(years * 365.25)))
        logger.info(
            "SP500DataBuilder: %s -> %s (%d yrs), batch=%d, dir=%s",
            self.start, self.end, years, batch_size, self.data_dir,
        )

    # ------------------------------------------------------------------ tickers
    def get_constituents(self) -> pd.DataFrame:
        """Scrape current S&P 500 constituents from Wikipedia.

        Returns a DataFrame: ticker (yfinance-normalized), name, sector.
        Uses requests + bs4's stdlib html.parser so no lxml/html5lib dependency
        is required.
        """
        logger.info("Fetching S&P 500 constituents from Wikipedia...")
        resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            # Fallback: first wikitable on the page.
            table = soup.find("table", {"class": "wikitable"})
        if table is None:
            raise RuntimeError("Could not locate the S&P 500 constituents table on Wikipedia")

        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 4 or tr.find("th"):  # skip header row(s)
                continue
            symbol = cells[0].get_text(strip=True)
            name = cells[1].get_text(strip=True)
            sector = cells[2].get_text(strip=True)
            if symbol:
                rows.append((symbol, name, sector))

        df = pd.DataFrame(rows, columns=["raw_symbol", "name", "sector"])
        # yfinance uses '-' where Wikipedia uses '.' (e.g. BRK.B -> BRK-B, BF.B -> BF-B).
        df["ticker"] = df["raw_symbol"].str.replace(".", "-", regex=False).str.upper()
        df = df.drop_duplicates(subset="ticker").reset_index(drop=True)
        logger.info("Found %d unique constituents", len(df))
        return df[["ticker", "name", "sector"]]

    # ----------------------------------------------------------------- download
    def _download_batch(self, tickers, attempt=1):
        """Download one batch as a wide multi-index frame. None on hard failure."""
        import yfinance as yf
        try:
            data = yf.download(
                tickers,
                start=self.start.isoformat(),
                end=(self.end + timedelta(days=1)).isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,   # keep BOTH close and adj_close
                threads=True,
                progress=False,
                timeout=30,
            )
            if data is None or data.empty:
                raise ValueError("empty frame")
            return data
        except Exception as e:  # noqa: BLE001 - we want to retry/skip anything
            logger.warning("Batch (attempt %d) failed: %s", attempt, e)
            if attempt < 2:
                time.sleep(3)
                return self._download_batch(tickers, attempt + 1)
            return None

    @staticmethod
    def _tidy_one(wide: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Extract a single ticker's columns from a (possibly multi-index) wide frame
        and return a tidy long slice. Empty frame if the ticker has no usable data."""
        # When a batch has >1 ticker, columns are a MultiIndex (ticker, field).
        # When only 1 ticker survives, yfinance may return a flat frame.
        if isinstance(wide.columns, pd.MultiIndex):
            if ticker not in wide.columns.get_level_values(0):
                return pd.DataFrame()
            sub = wide[ticker].copy()
        else:
            sub = wide.copy()

        rename = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        }
        sub = sub.rename(columns=rename)
        needed = ["open", "high", "low", "close", "adj_close", "volume"]
        if not set(needed).issubset(sub.columns):
            return pd.DataFrame()

        sub = sub[needed].dropna(how="all")
        # A row needs at least adj_close to be useful for returns.
        sub = sub.dropna(subset=["adj_close"])
        if sub.empty:
            return pd.DataFrame()

        sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
        sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.normalize()
        sub["ticker"] = ticker
        return sub[PANEL_COLUMNS]

    def build_panel(self, constituents: pd.DataFrame):
        """Download all tickers in batches and assemble the tidy panel.

        Returns (panel_df, report_dict).
        """
        tickers = constituents["ticker"].tolist()
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]
        logger.info("Downloading %d tickers in %d batch(es) of %d",
                    len(tickers), len(batches), self.batch_size)

        frames = []
        dropped = {}  # ticker -> reason
        fetched = set()

        for bi, batch in enumerate(batches, 1):
            logger.info("Batch %d/%d (%d tickers)...", bi, len(batches), len(batch))
            wide = self._download_batch(batch)
            if wide is None:
                for t in batch:
                    dropped[t] = "batch download failed"
                continue
            for t in batch:
                tidy = self._tidy_one(wide, t)
                if tidy.empty:
                    dropped[t] = "no usable rows returned"
                else:
                    frames.append(tidy)
                    fetched.add(t)
            time.sleep(1)  # be gentle between batches

        if not frames:
            raise RuntimeError("No data fetched for any ticker — aborting")

        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

        report = self._quality_report(panel, constituents, fetched, dropped)
        return panel, report

    # ------------------------------------------------------------------- report
    def _quality_report(self, panel, constituents, fetched, dropped) -> dict:
        """Compute the data-quality summary."""
        # Build a full date x ticker grid using the union of trading days actually seen,
        # so "% missing" is measured against the real trading calendar (not weekends).
        all_dates = panel["date"].drop_duplicates().sort_values()
        n_days = len(all_dates)

        counts = panel.groupby("ticker")["date"].nunique()
        missing_pct = ((n_days - counts) / n_days * 100).round(2)
        missing_pct = missing_pct.sort_values(ascending=False)

        return {
            "requested": len(constituents),
            "fetched": len(fetched),
            "dropped": dropped,
            "n_trading_days": n_days,
            "date_min": str(all_dates.min().date()),
            "date_max": str(all_dates.max().date()),
            "total_rows": len(panel),
            "missing_pct": missing_pct,
        }

    @staticmethod
    def print_report(report: dict):
        print("\n" + "=" * 60)
        print("DATA-QUALITY SUMMARY — S&P 500 panel (Phase 1)")
        print("=" * 60)
        print(f"Tickers requested : {report['requested']}")
        print(f"Tickers fetched   : {report['fetched']}")
        print(f"Tickers dropped   : {len(report['dropped'])}")
        print(f"Date range        : {report['date_min']} -> {report['date_max']}")
        print(f"Trading days      : {report['n_trading_days']}")
        print(f"Total panel rows  : {report['total_rows']:,}")

        mp = report["missing_pct"]
        print(f"\nMissing days per ticker (vs {report['n_trading_days']}-day calendar):")
        print(f"  mean   : {mp.mean():.2f}%")
        print(f"  median : {mp.median():.2f}%")
        print(f"  max    : {mp.max():.2f}%")
        worst = mp[mp > 0].head(15)
        if len(worst):
            print(f"\n  Top {len(worst)} most-incomplete tickers:")
            for tk, pct in worst.items():
                print(f"    {tk:<8} {pct:>6.2f}% missing")
        else:
            print("  (all tickers complete over the observed calendar)")

        if report["dropped"]:
            print(f"\nDropped tickers ({len(report['dropped'])}):")
            for tk, reason in report["dropped"].items():
                print(f"    {tk:<8} {reason}")
        print("=" * 60 + "\n")

    # -------------------------------------------------------------------- saving
    def save(self, panel: pd.DataFrame, constituents: pd.DataFrame):
        panel_path = self.data_dir / "sp500_panel.parquet"
        tickers_path = self.data_dir / "sp500_tickers.csv"
        panel.to_parquet(panel_path, index=False)
        constituents.to_csv(tickers_path, index=False)
        logger.info("Saved panel -> %s (%d rows)", panel_path, len(panel))
        logger.info("Saved ticker metadata -> %s", tickers_path)
        return panel_path, tickers_path


def main():
    builder = SP500DataBuilder(data_dir="data", years=7, batch_size=50)
    constituents = builder.get_constituents()
    panel, report = builder.build_panel(constituents)
    builder.save(panel, constituents)
    builder.print_report(report)


if __name__ == "__main__":
    main()
