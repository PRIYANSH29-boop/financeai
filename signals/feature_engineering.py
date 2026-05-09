"""
Feature engineering for stock price prediction.

Takes raw OHLCV (Open/High/Low/Close/Volume) data and creates
~30 features that capture price patterns, momentum, volatility,
and volume dynamics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates ML features from OHLCV stock data."""

    def __init__(self, prediction_horizon: int = 5):
        """
        Args:
            prediction_horizon: How many days ahead we predict.
                                5 = predict if price will be higher in 5 trading days.
        """
        self.horizon = prediction_horizon
        self.feature_cols = []

    def build(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Build features and target from raw OHLCV data.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume

        Returns:
            DataFrame with features + target column, ready for ML.
            Rows with insufficient history are dropped.
        """
        if df is None or len(df) < 100:
            logger.error(f"Need at least 100 rows of data, got {len(df) if df is not None else 0}")
            return None

        df = df.copy()

        # 1. RETURNS - the most fundamental feature
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)

        # 2. PRICE POSITION - where is price relative to recent history?
        df['price_vs_sma20'] = df['Close'] / df['Close'].rolling(20).mean()
        df['price_vs_sma50'] = df['Close'] / df['Close'].rolling(50).mean()
        df['price_vs_sma200'] = df['Close'] / df['Close'].rolling(200).mean()

        # Distance from 52-week high/low
        df['high_52w'] = df['Close'].rolling(252).max()
        df['low_52w'] = df['Close'].rolling(252).min()
        df['pct_from_high'] = (df['Close'] - df['high_52w']) / df['high_52w']
        df['pct_from_low'] = (df['Close'] - df['low_52w']) / df['low_52w']

        # 3. MOMENTUM (RSI - Relative Strength Index)
        df['rsi_14'] = self._rsi(df['Close'], 14)
        df['rsi_5'] = self._rsi(df['Close'], 5)

        # 4. MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['macd_diff_pct'] = df['macd_diff'] / df['Close']

        # 5. VOLATILITY
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_60d'] = df['returns_1d'].rolling(60).std()
        df['volatility_ratio'] = df['volatility_20d'] / df['volatility_60d']

        # ATR - Average True Range
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / df['Close']

        # 6. BOLLINGER BANDS
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid

        # 7. VOLUME
        df['volume_sma20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma20']
        df['volume_change'] = df['Volume'].pct_change()

        # OBV - On-Balance Volume
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_sma20'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma20']

        # 8. PRICE PATTERN
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['close_to_high'] = (df['High'] - df['Close']) / (df['High'] - df['Low'])

        # 9. THE TARGET - what we want to predict
        future_close = df['Close'].shift(-self.horizon)
        df['future_return'] = (future_close - df['Close']) / df['Close']
        df['target'] = (df['future_return'] > 0).astype(int)

        # 10. CLEAN UP
        feature_cols = [
            'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'pct_from_high', 'pct_from_low',
            'rsi_14', 'rsi_5',
            'macd', 'macd_signal', 'macd_diff', 'macd_diff_pct',
            'volatility_20d', 'volatility_60d', 'volatility_ratio',
            'atr_14', 'atr_pct',
            'bb_position', 'bb_width',
            'volume_ratio', 'volume_change',
            'obv_ratio',
            'high_low_pct', 'close_to_high',
        ]

        result = df[feature_cols + ['target', 'future_return', 'Close']].dropna()

        logger.info(f"Built {len(feature_cols)} features. Final dataset: {len(result)} rows.")

        self.feature_cols = feature_cols

        return result

    @staticmethod
    def _rsi(series, period=14):
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# Quick test when run directly
if __name__ == "__main__":
    import yfinance as yf
    logging.basicConfig(level=logging.INFO)

    print("Fetching 5 years of AAPL data for feature engineering test...")
    raw = yf.Ticker("AAPL").history(period="5y")

    fe = FeatureEngineer(prediction_horizon=5)
    features = fe.build(raw)

    if features is not None:
        print(f"\nDataset shape: {features.shape}")
        print(f"Number of features: {len(fe.feature_cols)}")
        print(f"\nTarget balance:")
        print(features['target'].value_counts())
        print(f"\nFirst 3 rows of features:")
        print(features[fe.feature_cols].head(3).round(3))
        print(f"\nMean future_return when target=1: {features[features['target']==1]['future_return'].mean():.4f}")
        print(f"Mean future_return when target=0: {features[features['target']==0]['future_return'].mean():.4f}")