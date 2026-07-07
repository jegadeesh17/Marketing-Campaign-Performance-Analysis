"""Schema and shape tests for campaign inference helpers."""

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.inference import CHANNELS, build_campaign_row


def test_build_campaign_row_shapes():
    df_reg, df_cls = build_campaign_row({"brand": "nykaa"})
    assert len(df_reg) == 1
    assert len(df_cls) == 1
    assert "acquisition_cost" in df_reg.columns
    assert "revenue" not in df_reg.columns


def test_channel_flags_present():
    df_reg, _ = build_campaign_row({"channels": ["Instagram", "Google"]})
    for ch in CHANNELS:
        assert f"channel_{ch.lower()}" in df_reg.columns
    assert df_reg["channel_instagram"].iloc[0] == 1
    assert df_reg["channel_google"].iloc[0] == 1
    assert df_reg["channel_youtube"].iloc[0] == 0


def test_derived_metrics_computed():
    df_reg, _ = build_campaign_row(
        {"impressions": 10000, "clicks": 500, "leads": 100, "conversions": 50, "acquisition_cost": 200}
    )
    assert df_reg["ctr"].iloc[0] == 0.05
    assert df_reg["conversion_rate"].iloc[0] == 0.1
    assert df_reg["cpl"].iloc[0] == 2.0


def test_month_cyclical_encoding():
    df_reg, _ = build_campaign_row({"month": 3})
    assert "month_sin" in df_reg.columns
    assert "month_cos" in df_reg.columns
    assert -1 <= df_reg["month_sin"].iloc[0] <= 1


def test_zero_safe_divisions():
    df_reg, _ = build_campaign_row({"impressions": 0, "clicks": 0, "leads": 0})
    assert df_reg["ctr"].iloc[0] == 0
    assert df_reg["conversion_rate"].iloc[0] == 0
    assert df_reg["cpl"].iloc[0] == 0


def test_categorical_defaults():
    df_reg, _ = build_campaign_row({})
    for col in ("campaign_type", "target_audience", "language", "customer_segment", "brand"):
        assert col in df_reg.columns
        assert isinstance(df_reg[col].iloc[0], str)


def test_classifier_row_is_copy():
    df_reg, df_cls = build_campaign_row({"brand": "purplle"})
    df_cls["revenue"] = 99999.0
    assert "revenue" not in df_reg.columns
    assert df_cls["revenue"].iloc[0] == 99999.0
