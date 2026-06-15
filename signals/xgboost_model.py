"""
XGBoost classifier for stock direction prediction.

Trains a gradient boosted tree ensemble to predict whether
a stock will be higher in N days. Uses chronological train/test
split to avoid look-ahead bias.

Algorithm: XGBoost (Extreme Gradient Boosting)
  - Builds many small decision trees sequentially
  - Each tree corrects the residual errors of previous trees
  - Final prediction = weighted sum of all trees
  - Regularization built in (prevents overfitting)
"""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class XGBoostSignalModel:
    """XGBoost model for predicting stock direction."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_cols: Optional[list] = None
        self.metrics: dict = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: list,
        params: Optional[dict] = None,
    ) -> None:
        """
        Lean training method - just fits the model, no evaluation.
        Use this inside backtesters where evaluation is done externally.
        """
        self.feature_cols = feature_cols

        default_params = {
            "n_estimators": 200,
            "max_depth": 4,
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
            "n_jobs": -1,
            "verbosity": 0,
        }
        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X_train[feature_cols], y_train, verbose=False)

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        test_size: float = 0.2,
        params: Optional[dict] = None,
        horizon: int = 5,
    ) -> dict:
        """
        Train XGBoost on engineered features with chronological split.

        Args:
            df: DataFrame from FeatureEngineer.build()
            feature_cols: list of column names to use as features
            test_size: fraction of data held out for testing (0.2 = 20%)
            params: optional XGBoost hyperparameters override

        Returns:
            dict of evaluation metrics on the test set
        """
        self.feature_cols = feature_cols

        # Chronological split (no shuffling — finance must respect time)
        split_idx = int(len(df) * (1 - test_size))
        # Purge the last `horizon` training rows: their labels (Close.shift(-horizon))
        # peek into the test set and would leak look-ahead information.
        train_df = df.iloc[: split_idx - horizon]
        test_df = df.iloc[split_idx:]

        logger.info(
            f"Train: {len(train_df)} rows ({train_df.index.min().date()} → "
            f"{train_df.index.max().date()})"
        )
        logger.info(
            f"Test:  {len(test_df)} rows ({test_df.index.min().date()} → "
            f"{test_df.index.max().date()})"
        )

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test = test_df[feature_cols]
        y_test = test_df["target"]

        # Use the lean fit method
        self.fit(X_train, y_train, feature_cols, params)

        # Evaluate on test set
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        self.metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else None,
            "test_size": len(y_test),
            "baseline_accuracy": max(y_test.mean(), 1 - y_test.mean()),
        }

        self._print_report(y_test, y_pred, y_pred_proba)

        return self.metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for new data. Returns probability of target=1."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")
        return self.model.predict_proba(features[self.feature_cols])[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        """Return features ranked by importance."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importance = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, name: str = "xgboost_signal") -> None:
        """Save the trained model and metadata."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        model_path = self.model_dir / f"{name}.json"
        meta_path = self.model_dir / f"{name}_meta.json"

        self.model.save_model(model_path)
        with open(meta_path, "w") as f:
            json.dump(
                {"feature_cols": self.feature_cols, "metrics": self.metrics},
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Saved model to {model_path}")

    def load(self, name: str = "xgboost_signal") -> None:
        """Load a previously trained model."""
        model_path = self.model_dir / f"{name}.json"
        meta_path = self.model_dir / f"{name}_meta.json"

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self.metrics = meta["metrics"]
        logger.info(f"Loaded model from {model_path}")

    @staticmethod
    def _print_report(y_test, y_pred, y_pred_proba):
        """Pretty print evaluation results."""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION (on held-out test set)")
        print("=" * 70)

        print(f"\nTest set size: {len(y_test)} rows")
        print(f"Test target balance: {y_test.value_counts().to_dict()}")
        baseline = max(y_test.mean(), 1 - y_test.mean())
        print(f"Baseline (always predict majority): {baseline:.3f}")

        print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}  (when we say UP, how often right)")
        print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}  (of all UP days, how many we caught)")
        print(f"F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}  (harmonic mean of precision/recall)")
        if len(set(y_test)) > 1:
            print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.3f}  (overall ranking quality)")
        else:
            print(f"ROC-AUC:   undefined  (test set has only one class)")

        print("\nConfusion matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(f"                 Predicted DOWN   Predicted UP")
        print(f"  Actually DOWN: {cm[0][0]:>14}   {cm[0][1]:>11}")
        print(f"  Actually UP:   {cm[1][0]:>14}   {cm[1][1]:>11}")

        print("\nFull classification report:")
        print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"], zero_division=0))


# ── QUICK TEST WHEN RUN DIRECTLY ──
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import yfinance as yf
    from signals.feature_engineering import FeatureEngineer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("Fetching 5y AAPL data...")
    raw = yf.Ticker("AAPL").history(period="5y")

    print("Engineering features...")
    fe = FeatureEngineer(prediction_horizon=5)
    dataset = fe.build(raw)

    print("Training XGBoost model...\n")
    model = XGBoostSignalModel()
    metrics = model.train(dataset, fe.feature_cols)

    print("\n" + "=" * 70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 70)
    print(model.feature_importance().head(10).to_string(index=False))

    model.save()
    print("\n✅ Model trained and saved.")