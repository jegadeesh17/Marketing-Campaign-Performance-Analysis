"""Build campaign feature rows for model inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

CHANNELS = ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"]

ORDERED_COLUMNS = [
    "campaign_type",
    "target_audience",
    "language",
    "customer_segment",
    "brand",
    "impressions",
    "clicks",
    "leads",
    "conversions",
    "engagement_score",
    "month_sin",
    "month_cos",
    "ctr",
    "conversion_rate",
    "cpl",
    "channel_youtube",
    "channel_instagram",
    "channel_google",
    "channel_whatsapp",
    "channel_email",
    "channel_facebook",
    "acquisition_cost",
]


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


def build_batch_campaign_rows(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct vectorized multi-row regression and classification DataFrames for batch inference."""
    if not payloads:
        df_empty = pd.DataFrame(columns=ORDERED_COLUMNS)
        return df_empty, df_empty.copy()

    df = pd.DataFrame(payloads).copy()

    # Categorical fields imputation and normalization
    for col, default_val in [
        ("campaign_type", "Paid Ads"),
        ("target_audience", "Youth"),
        ("language", "English"),
        ("customer_segment", "Premium Shoppers"),
    ]:
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val).replace("", default_val).astype(str)

    if "brand" not in df.columns:
        df["brand"] = "nykaa"
    else:
        df["brand"] = df["brand"].fillna("nykaa").replace("", "nykaa").astype(str).str.strip().str.lower()
        df["brand"] = df["brand"].replace("", "nykaa")

    # Numerical fields imputation and casting
    for col, default_val in [
        ("impressions", 50000.0),
        ("clicks", 4000.0),
        ("leads", 1500.0),
        ("conversions", 500.0),
        ("engagement_score", 15.0),
        ("acquisition_cost", 250.0),
    ]:
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val).astype(float)

    if "month" not in df.columns:
        month_series = pd.Series(5.0, index=df.index)
    else:
        month_series = pd.to_numeric(df["month"], errors="coerce").fillna(5.0).astype(float)

    # Cyclical month sin/cos encodings
    df["month_sin"] = np.sin(2.0 * np.pi * month_series / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month_series / 12.0)

    # Vectorized zero-safe ratio derivations
    impressions = df["impressions"].to_numpy(dtype=float)
    clicks = df["clicks"].to_numpy(dtype=float)
    leads = df["leads"].to_numpy(dtype=float)
    conversions = df["conversions"].to_numpy(dtype=float)
    acquisition_cost = df["acquisition_cost"].to_numpy(dtype=float)

    ctr = np.zeros(len(df), dtype=float)
    np.divide(clicks, impressions, out=ctr, where=(impressions > 0))
    df["ctr"] = np.nan_to_num(ctr, nan=0.0, posinf=0.0, neginf=0.0)

    conversion_rate = np.zeros(len(df), dtype=float)
    np.divide(conversions, clicks, out=conversion_rate, where=(clicks > 0))
    df["conversion_rate"] = np.nan_to_num(conversion_rate, nan=0.0, posinf=0.0, neginf=0.0)

    cpl = np.zeros(len(df), dtype=float)
    np.divide(acquisition_cost, leads, out=cpl, where=(leads > 0))
    df["cpl"] = np.nan_to_num(cpl, nan=0.0, posinf=0.0, neginf=0.0)

    # Vectorized multi-channel one-hot flags
    channels = df["channels"].copy() if "channels" in df.columns else pd.Series(index=df.index, dtype=object)
    empty_mask = channels.isna() | (channels.str.len() == 0)
    channels = channels.where(~empty_mask, pd.Series([["Instagram", "Google"]] * len(df), index=df.index))
    exploded = channels.explode().astype(str).str.strip().str.lower()

    for ch in CHANNELS:
        col = f"channel_{ch.lower()}"
        df[col] = 0
        matched_idx = exploded[exploded == ch.lower()].index.unique()
        df.loc[matched_idx, col] = 1
        df[col] = df[col].astype(int)

    df_reg = df[ORDERED_COLUMNS].copy()
    df_cls = df_reg.copy()
    return df_reg, df_cls
