"""Build campaign feature rows for model inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

CHANNELS = ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"]


def build_campaign_row(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    impressions = float(payload.get("impressions", 50000))
    clicks = float(payload.get("clicks", 4000))
    leads = float(payload.get("leads", 1500))
    conversions = float(payload.get("conversions", 500))
    acquisition_cost = float(payload.get("acquisition_cost", 250.0))
    month = int(payload.get("month", 5))
    selected_channels = payload.get("channels", ["Instagram", "Google"])

    ctr = clicks / impressions if impressions > 0 else 0
    conversion_rate = conversions / clicks if clicks > 0 else 0
    cpl = acquisition_cost / leads if leads > 0 else 0
    month_sin = math.sin(2 * math.pi * float(month) / 12)
    month_cos = math.cos(2 * math.pi * float(month) / 12)

    row = {
        "campaign_type": payload.get("campaign_type", "Paid Ads"),
        "target_audience": payload.get("target_audience", "Youth"),
        "language": payload.get("language", "English"),
        "customer_segment": payload.get("customer_segment", "Premium Shoppers"),
        "brand": payload.get("brand", "nykaa"),
        "impressions": impressions,
        "clicks": clicks,
        "leads": leads,
        "conversions": conversions,
        "engagement_score": float(payload.get("engagement_score", 15.0)),
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
