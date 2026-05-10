"""
SHAP-based explainability for the XGBoost signal model.

Algorithm: SHAP (SHapley Additive exPlanations)
  - Game theory based: treats each feature as a "player"
  - Fairly distributes the prediction's deviation from average
    among all features
  - For tree models, uses TreeSHAP for fast exact computation

For every prediction, SHAP tells us:
  - Which features pushed the prediction UP
  - Which features pushed it DOWN
  - By exactly how much (additive contributions)
"""

import sys
import warnings
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainability for XGBoost signal models."""

    def __init__(self, model: xgb.XGBClassifier, feature_cols: list):
        """
        Args:
            model: a trained XGBoost classifier
            feature_cols: ordered list of feature column names
        """
        self.model = model
        self.feature_cols = feature_cols
        # TreeExplainer is the fast SHAP variant for tree-based models
        self.explainer = shap.TreeExplainer(model)

    def explain_one(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Explain a single prediction.

        Args:
            features: a 1-row DataFrame with feature_cols

        Returns:
            DataFrame with columns [feature, value, shap_value, direction]
            sorted by absolute SHAP value (most impactful first)
        """
        if len(features) != 1:
            raise ValueError(f"Expected 1 row, got {len(features)}")

        # SHAP values for the positive class (probability of UP)
        shap_vals = self.explainer.shap_values(features[self.feature_cols])

        # Some SHAP versions return list, some return array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        shap_vals = shap_vals[0]  # extract single row

        # Build explanation table
        result = pd.DataFrame({
            "feature": self.feature_cols,
            "value": features[self.feature_cols].iloc[0].values,
            "shap_value": shap_vals,
        })
        result["direction"] = result["shap_value"].apply(
            lambda x: "↑ bullish" if x > 0 else ("↓ bearish" if x < 0 else "neutral")
        )
        result["abs_impact"] = result["shap_value"].abs()
        result = result.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")
        result = result.reset_index(drop=True)

        return result

    def global_importance(self, features: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Calculate global feature importance based on mean |SHAP| across many predictions.

        This is the SHAP-based answer to 'which features matter most overall?'
        It's more reliable than XGBoost's built-in feature_importances_.
        """
        shap_vals = self.explainer.shap_values(features[self.feature_cols])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        result = pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return result.head(top_n)

    def text_summary(self, features: pd.DataFrame, top_n: int = 5) -> str:
        """
        Produce a human-readable text explanation for a single prediction.

        This is the format we'll feed to the LLM later — clean, structured,
        cite-friendly.
        """
        explanation = self.explain_one(features)
        prediction = self.model.predict_proba(features[self.feature_cols])[0, 1]

        lines = [
            f"Model probability of UP: {prediction:.1%}",
            f"Top {top_n} drivers of this prediction:",
            "",
        ]

        for _, row in explanation.head(top_n).iterrows():
            lines.append(
                f"  • {row['feature']:25s} = {row['value']:8.4f}  "
                f"→ SHAP {row['shap_value']:+.4f}  ({row['direction']})"
            )

        # Aggregate impact direction
        bullish = explanation[explanation["shap_value"] > 0]["shap_value"].sum()
        bearish = explanation[explanation["shap_value"] < 0]["shap_value"].sum()

        lines.extend([
            "",
            f"Total bullish contribution: {bullish:+.4f}",
            f"Total bearish contribution: {bearish:+.4f}",
            f"Net:                        {bullish + bearish:+.4f}",
        ])

        return "\n".join(lines)


# ── QUICK TEST ──
if __name__ == "__main__":
    import yfinance as yf
    from signals.feature_engineering import FeatureEngineer
    from signals.xgboost_model import XGBoostSignalModel

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 1. Load data and train a model
    print("Fetching 5y AAPL data...")
    raw = yf.Ticker("AAPL").history(period="5y")

    print("Engineering features...")
    fe = FeatureEngineer(prediction_horizon=5)
    dataset = fe.build(raw)

    print("Training model...")
    model = XGBoostSignalModel()
    model.train(dataset, fe.feature_cols, test_size=0.2)

    # 2. Build the explainer
    print("\nBuilding SHAP explainer...")
    explainer = SHAPExplainer(model.model, model.feature_cols)

    # 3. Global importance
    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE (mean |SHAP|)")
    print("=" * 70)
    global_imp = explainer.global_importance(dataset, top_n=15)
    print(global_imp.to_string(index=False))

    # 4. Explain the latest prediction
    print("\n" + "=" * 70)
    print("EXPLANATION FOR THE LATEST AAPL PREDICTION")
    print("=" * 70)
    latest = dataset.iloc[[-1]]  # last row, keep as DataFrame
    explanation = explainer.text_summary(latest, top_n=8)
    print(explanation)

    print("\n✅ SHAP explainer ready.")