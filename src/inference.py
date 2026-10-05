"""Build campaign feature rows for model inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

CHANNELS = ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"]


def build_campaign_row(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct single-row regression and classification DataFrames with derived metrics."""
    impressions = float(payload.get("impressions") if payload.get("impressions") is not None else 50000.0)
    clicks = float(payload.get("clicks") if payload.get("clicks") is not None else 4000.0)
    leads = float(payload.get("leads") if payload.get("leads") is not None else 1500.0)
    conversions = float(payload.get("conversions") if payload.get("conversions") is not None else 500.0)
    acquisition_cost = float(payload.get("acquisition_cost") if payload.get("acquisition_cost") is not None else 250.0)
    month = int(payload.get("month") if payload.get("month") is not None else 5)
    selected_channels = payload.get("channels") or ["Instagram", "Google"]

    ctr = float(clicks / impressions) if impressions > 0 else 0.0
    if math.isnan(ctr) or math.isinf(ctr):
        ctr = 0.0

    conversion_rate = float(conversions / clicks) if clicks > 0 else 0.0
    if math.isnan(conversion_rate) or math.isinf(conversion_rate):
        conversion_rate = 0.0

    cpl = float(acquisition_cost / leads) if leads > 0 else 0.0
    if math.isnan(cpl) or math.isinf(cpl):
        cpl = 0.0

    month_sin = math.sin(2 * math.pi * float(month) / 12)
    month_cos = math.cos(2 * math.pi * float(month) / 12)

    brand = str(payload.get("brand") or "nykaa").strip().lower()

    row = {
        "campaign_type": payload.get("campaign_type") or "Paid Ads",
        "target_audience": payload.get("target_audience") or "Youth",
        "language": payload.get("language") or "English",
        "customer_segment": payload.get("customer_segment") or "Premium Shoppers",
        "brand": brand,
        "impressions": impressions,
        "clicks": clicks,
        "leads": leads,
        "conversions": conversions,
        "engagement_score": float(payload.get("engagement_score") if payload.get("engagement_score") is not None else 15.0),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "ctr": ctr,
        "conversion_rate": conversion_rate,
        "cpl": cpl,
    }
    for ch in CHANNELS:
        row[f"channel_{ch.lower()}"] = 1 if ch in selected_channels else 0

    df_reg = pd.DataFrame([row])
    df_reg["acquisition_cost"] = acquisition_cost
    df_cls = df_reg.copy()
    return df_reg, df_cls
