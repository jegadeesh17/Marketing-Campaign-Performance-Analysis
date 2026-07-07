"""Export model metrics to reports/evaluation.md."""

from __future__ import annotations

import json
import os
import sys

import joblib
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_clean_data

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPORT = os.path.join(ROOT, "reports", "evaluation.md")
METRICS = os.path.join(ROOT, "reports", "metrics.json")


def main() -> None:
    reg_path = os.path.join(ROOT, "models", "revenue_regressor.joblib")
    cls_path = os.path.join(ROOT, "models", "profit_classifier.joblib")
    if not os.path.exists(reg_path) or not os.path.exists(cls_path):
        raise FileNotFoundError("Train models first: python src/train_models.py")

    df = load_and_clean_data()
    cat_features = ['campaign_type', 'target_audience', 'language', 'customer_segment', 'brand']
    num_features_base = ['impressions', 'clicks', 'leads', 'conversions', 'engagement_score', 'month_sin', 'month_cos', 'ctr', 'conversion_rate', 'cpl']
    channel_features = [col for col in df.columns if col.startswith('channel_') and col != 'channel_used']

    reg = joblib.load(reg_path)
    cls = joblib.load(cls_path)

    X_reg = df[cat_features + num_features_base + channel_features + ['acquisition_cost']]
    y_reg = df['revenue']
    _, X_test_r, _, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    preds_r = reg.predict(X_test_r)
    r2 = r2_score(y_test_r, preds_r)
    rmse = mean_squared_error(y_test_r, preds_r) ** 0.5

    X_cls = df[cat_features + num_features_base + channel_features + ['revenue', 'acquisition_cost']]
    y_cls = df['profit_flag']
    _, X_test_c, _, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    preds_c = cls.predict(X_test_c)
    acc = accuracy_score(y_test_c, preds_c)
    report = classification_report(y_test_c, preds_c, output_dict=True)

    metrics = {
        "regression": {"r2": round(float(r2), 4), "rmse": round(float(rmse), 2)},
        "classification": {
            "accuracy": round(float(acc), 4),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        },
        "split": "80/20 holdout",
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# Marketing Campaign Analysis — Evaluation Report",
        "",
        "## Regression (Revenue Forecast)",
        f"- R²: {metrics['regression']['r2']}",
        f"- RMSE: {metrics['regression']['rmse']}",
        "",
        "## Classification (Profitability)",
        f"- Accuracy: {metrics['classification']['accuracy']}",
        f"- Weighted F1: {metrics['classification']['weighted_f1']}",
        "",
        "## Notes",
        "- Classification uses revenue + acquisition_cost as features after revenue forecast (leakage-aware pipeline).",
        "- Regenerate: `python src/train_models.py && python scripts/export_evaluation.py`",
    ]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
