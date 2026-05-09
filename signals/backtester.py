"""
Walk-forward backtester for the XGBoost signal model.
Memory-efficient version for low-RAM hardware.

Algorithm:
  1. Iterate through time, retraining the model periodically
  2. At each retrain point, train on all historical data
  3. Predict on the next out-of-sample window
  4. Track predictions + actual outcomes + simulated P&L
"""

import sys
import gc
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from signals.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

# Force flush after every log line so we see progress in real-time
for h in logging.getLogger().handlers:
    h.flush = lambda: None


# Lightweight params for low-RAM hardware
LIGHT_PARAMS = {
    "n_estimators": 100,        # was 200
    "max_depth": 3,              # was 4
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 2,                 # was -1 (all cores) — limit to 2
    "verbosity": 0,
}


class WalkForwardBacktester:
    """Walk-forward backtest, memory-efficient."""

    def __init__(
        self,
        initial_train_size: int = 504,
        retrain_freq: int = 42,        # was 21 — retrain every 2 months
        prediction_horizon: int = 5,
        results_dir: str = "data",
    ):
        self.initial_train_size = initial_train_size
        self.retrain_freq = retrain_freq
        self.horizon = prediction_horizon
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run walk-forward backtest."""
        feature_cols = [
            c for c in df.columns
            if c not in ("target", "future_return", "Close")
        ]

        n = len(df)
        if n < self.initial_train_size + self.retrain_freq:
            raise ValueError(
                f"Need at least {self.initial_train_size + self.retrain_freq} "
                f"rows, got {n}"
            )

        # Estimate total iterations for progress display
        total_iters = (n - self.initial_train_size - self.horizon) // self.retrain_freq + 1
        print(f"Total retrains expected: ~{total_iters}", flush=True)

        results = []
        retrain_count = 0

        i = self.initial_train_size
        while i < n - self.horizon:
            window_end = min(i + self.retrain_freq, n - self.horizon)

            train = df.iloc[:i]
            test = df.iloc[i:window_end]

            # Train using XGBoost directly (lighter than wrapper class)
            model = xgb.XGBClassifier(**LIGHT_PARAMS)
            model.fit(train[feature_cols], train["target"], verbose=False)
            retrain_count += 1

            # Predict
            probas = model.predict_proba(test[feature_cols])[:, 1]
            preds = (probas >= 0.5).astype(int)

            # Record
            for date, prob, pred, actual, fwd_ret in zip(
                test.index,
                probas,
                preds,
                test["target"].values,
                test["future_return"].values,
            ):
                results.append({
                    "date": date,
                    "probability": prob,
                    "prediction": pred,
                    "actual": actual,
                    "future_return": fwd_ret,
                    "retrain_id": retrain_count,
                })

            # Progress log — flushed immediately
            print(
                f"  [{retrain_count}/{total_iters}] "
                f"Trained on {len(train)} rows, "
                f"predicted {test.index[0].date()} → {test.index[-1].date()}",
                flush=True,
            )

            # Free memory before next iteration
            del model
            gc.collect()

            i = window_end

        results_df = pd.DataFrame(results).set_index("date")
        results_df.to_csv(self.results_dir / "backtest_results.csv")
        print(f"\n✅ Saved {len(results_df)} predictions to backtest_results.csv", flush=True)

        return results_df

    def evaluate(self, results: pd.DataFrame) -> dict:
        acc = accuracy_score(results["actual"], results["prediction"])
        try:
            auc = roc_auc_score(results["actual"], results["probability"])
        except ValueError:
            auc = None

        strategy_returns = np.where(
            results["prediction"] == 1,
            results["future_return"],
            0,
        )
        bh_returns = results["future_return"].values

        total_strategy_return = strategy_returns.sum() / self.horizon
        total_bh_return = bh_returns.sum() / self.horizon

        if strategy_returns.std() > 0:
            sharpe = (
                np.sqrt(252 / self.horizon)
                * strategy_returns.mean()
                / strategy_returns.std()
            )
        else:
            sharpe = 0.0

        cumulative = (1 + strategy_returns / self.horizon).cumprod()
        rolling_max = pd.Series(cumulative).cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        long_trades = results[results["prediction"] == 1]
        hit_rate_long = (
            (long_trades["actual"] == 1).mean()
            if len(long_trades) > 0
            else 0.0
        )

        return {
            "n_predictions": len(results),
            "n_long_signals": len(long_trades),
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "long_signal_hit_rate": round(hit_rate_long, 4),
            "total_strategy_return": round(float(total_strategy_return), 4),
            "total_buy_hold_return": round(float(total_bh_return), 4),
            "sharpe_ratio_annualized": round(float(sharpe), 4),
            "max_drawdown": round(float(max_dd), 4),
        }

    @staticmethod
    def print_report(metrics: dict):
        print("\n" + "=" * 70)
        print("WALK-FORWARD BACKTEST RESULTS")
        print("=" * 70)
        print(f"\nTotal predictions:           {metrics['n_predictions']}")
        print(f"Times we said 'UP':          {metrics['n_long_signals']}")
        print(f"\nAccuracy:                    {metrics['accuracy']:.3f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC-AUC:                     {metrics['roc_auc']:.3f}")
        print(f"Hit rate when predicting UP: {metrics['long_signal_hit_rate']:.3f}")

        print(f"\nStrategy total return:       {metrics['total_strategy_return']*100:+.2f}%")
        print(f"Buy & hold total return:     {metrics['total_buy_hold_return']*100:+.2f}%")
        diff = metrics['total_strategy_return'] - metrics['total_buy_hold_return']
        print(f"Strategy vs B&H difference:  {diff*100:+.2f}%")

        print(f"\nSharpe ratio (annualized):   {metrics['sharpe_ratio_annualized']:.2f}")
        print(f"Max drawdown:                {metrics['max_drawdown']*100:+.2f}%")
        print("=" * 70)


if __name__ == "__main__":
    import yfinance as yf

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("Fetching 5y AAPL data...", flush=True)
    raw = yf.Ticker("AAPL").history(period="5y")

    print("Engineering features...", flush=True)
    fe = FeatureEngineer(prediction_horizon=5)
    dataset = fe.build(raw)
    print(f"Dataset: {len(dataset)} rows", flush=True)

    print("\nStarting walk-forward backtest...", flush=True)
    bt = WalkForwardBacktester(
        initial_train_size=504,
        retrain_freq=42,
        prediction_horizon=5,
    )
    results = bt.run(dataset)

    metrics = bt.evaluate(results)
    bt.print_report(metrics)