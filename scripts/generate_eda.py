"""Generate EDA plots for marketing campaign data."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_clean_data

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "eda")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_clean_data()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["roi"], kde=True, bins=40)
    plt.title("ROI Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roi_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 8))
    numeric = df[["impressions", "clicks", "conversions", "revenue", "acquisition_cost", "roi"]]
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="campaign_type", y="roi")
    plt.xticks(rotation=30, ha="right")
    plt.title("ROI by Campaign Type")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roi_by_campaign_type.png"))
    plt.close()
    print(f"Saved 3 EDA plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
