"""
Data fetcher for stock market data.
Pulls price history, computes technical indicators, fetches fundamentals.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging

# Set up logging - real engineers log everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and processes stock market data from yfinance."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        logger.info(f"DataFetcher initialized. Data directory: {self.data_dir}")

    def fetch(self, ticker, period="1y"):
        """
        Fetch stock data for a given ticker.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            dict with 'prices' DataFrame, 'fundamentals' dict, 'summary' dict
            None if fetch failed
        """
        ticker = ticker.upper().strip()
        logger.info(f"Fetching data for {ticker} (period: {period})")

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                logger.error(f"No data returned for {ticker}")
                return None

            # Validate data quality
            df = self._validate_data(df, ticker)

            # Compute technical indicators
            df = self._compute_indicators(df)

            # Get fundamentals
            fundamentals = self._fetch_fundamentals(stock, ticker)

            # Build summary
            summary = self._build_summary(ticker, df, fundamentals)

            # Save to disk
            self._save(ticker, df, fundamentals, summary)

            return {
                'prices': df,
                'fundamentals': fundamentals,
                'summary': summary
            }

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return None

    def _validate_data(self, df, ticker):
        """Flag anomalies and clean obvious bad data."""
        # Calculate price changes
        df['price_change'] = df['Close'].pct_change()

        # Flag suspicious moves (>15% in a day usually = data error or major event)
        df['anomaly'] = df['price_change'].abs() > 0.15

        anomalies = df[df['anomaly']]
        if len(anomalies) > 0:
            logger.warning(f"{ticker}: {len(anomalies)} anomalous price moves detected")
            for idx in anomalies.index:
                logger.warning(f"  {idx.date()}: {df.loc[idx, 'price_change']*100:.1f}% move")

        # Drop rows with missing critical data
        before = len(df)
        df = df.dropna(subset=['Close', 'Volume'])
        if before > len(df):
            logger.warning(f"{ticker}: dropped {before - len(df)} rows with missing data")

        return df

    def _compute_indicators(self, df):
        """Compute technical indicators."""
        # RSI
        df['RSI'] = self._rsi(df['Close'])

        # MACD
        df['MACD'], df['MACD_signal'] = self._macd(df['Close'])

        # Bollinger Bands
        df['BB_upper'], df['BB_lower'] = self._bollinger(df['Close'])

        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        # Volume
        df['Volume_avg'] = df['Volume'].rolling(20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_avg']

        return df

    @staticmethod
    def _rsi(series, period=14):
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(series, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence."""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    @staticmethod
    def _bollinger(series, period=20, std_dev=2):
        """Bollinger Bands."""
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        return sma + (std_dev * std), sma - (std_dev * std)

    def _fetch_fundamentals(self, stock, ticker):
        """Fetch company fundamentals."""
        try:
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'market_cap': info.get('marketCap'),
                'revenue_growth': info.get('revenueGrowth'),
                'profit_margins': info.get('profitMargins'),
                'debt_to_equity': info.get('debtToEquity'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
            }
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {ticker}: {e}")
            return {}

    def _build_summary(self, ticker, df, fundamentals):
        """Build a clean summary of the latest state."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        price_change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

        def safe_round(value, decimals=2):
            """Round only if value is a number."""
            if pd.isna(value) or value is None:
                return None
            return round(float(value), decimals)

        bb_pos = None
        if not pd.isna(latest['BB_upper']) and not pd.isna(latest['BB_lower']):
            bb_pos = safe_round(
                (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
            )

        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'current_price': safe_round(latest['Close']),
            'price_change_pct': safe_round(price_change_pct),
            'rsi': safe_round(latest['RSI']),
            'macd': safe_round(latest['MACD'], 4),
            'macd_signal': safe_round(latest['MACD_signal'], 4),
            'macd_above_signal': bool(latest['MACD'] > latest['MACD_signal']) if not pd.isna(latest['MACD']) else None,
            'above_sma20': bool(latest['Close'] > latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
            'above_sma50': bool(latest['Close'] > latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
            'volume_ratio': safe_round(latest['Volume_ratio']),
            'bb_position': bb_pos,
            'fundamentals': fundamentals
        }

    def _save(self, ticker, df, fundamentals, summary):
        """Save everything to disk."""
        df.to_csv(self.data_dir / f'{ticker}_prices.csv')
        with open(self.data_dir / f'{ticker}_fundamentals.json', 'w') as f:
            json.dump(fundamentals, f, indent=2, default=str)
        with open(self.data_dir / f'{ticker}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved data for {ticker} to {self.data_dir}")


# Quick test when run directly
if __name__ == "__main__":
    fetcher = DataFetcher()
    ticker = input("Ticker: ").upper().strip()
    result = fetcher.fetch(ticker)

    if result:
        print("\n--- SUMMARY ---")
        print(json.dumps(result['summary'], indent=2, default=str))
    else:
        print("Failed to fetch data.")