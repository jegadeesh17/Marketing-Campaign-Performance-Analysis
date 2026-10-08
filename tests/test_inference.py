"""Schema and shape tests for campaign inference helpers."""

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.inference import CHANNELS, build_batch_campaign_rows, build_campaign_row


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


def test_zero_safe_ctr_ac_err_08():
    df_reg, _ = build_campaign_row({"impressions": 0, "clicks": 0})
    assert df_reg["ctr"].iloc[0] == 0.0
    assert not pd.isna(df_reg["ctr"].iloc[0])


def test_zero_safe_conversion_rate_ac_err_09():
    df_reg, _ = build_campaign_row({"clicks": 0, "conversions": 0})
    assert df_reg["conversion_rate"].iloc[0] == 0.0
    assert not pd.isna(df_reg["conversion_rate"].iloc[0])


def test_zero_safe_cpl_ac_err_10():
    df_reg, _ = build_campaign_row({"leads": 0, "acquisition_cost": 250.0})
    assert df_reg["cpl"].iloc[0] == 0.0
    assert not pd.isna(df_reg["cpl"].iloc[0])


def test_brand_normalization_in_inference():
    df_reg, _ = build_campaign_row({"brand": "Nykaa"})
    assert df_reg["brand"].iloc[0] == "nykaa"
    df_reg2, _ = build_campaign_row({"brand": "  PURPLLE  "})
    assert df_reg2["brand"].iloc[0] == "purplle"


def test_build_batch_campaign_rows_shapes():
    payloads = [
        {"brand": "nykaa", "impressions": 10000.0},
        {"brand": "purplle", "clicks": 200.0},
        {"brand": "tira", "channels": ["WhatsApp"]},
    ]
    df_reg, df_cls = build_batch_campaign_rows(payloads)
    assert len(df_reg) == 3
    assert len(df_cls) == 3
    single_reg, _ = build_campaign_row({})
    assert df_reg.columns.tolist() == single_reg.columns.tolist()
    assert df_cls.columns.tolist() == single_reg.columns.tolist()
    assert "acquisition_cost" in df_reg.columns
    assert "revenue" not in df_reg.columns


def test_build_batch_campaign_rows_empty():
    df_reg, df_cls = build_batch_campaign_rows([])
    assert len(df_reg) == 0
    assert len(df_cls) == 0
    single_reg, _ = build_campaign_row({})
    assert df_reg.columns.tolist() == single_reg.columns.tolist()
    assert df_cls.columns.tolist() == single_reg.columns.tolist()


def test_build_batch_campaign_rows_equivalence():
    payloads = [
        {"brand": "nykaa", "impressions": 50000, "clicks": 4000, "leads": 1500, "conversions": 500, "acquisition_cost": 250},
        {"brand": "purplle", "campaign_type": "Influencer", "target_audience": "Working Women", "month": 11, "channels": ["YouTube", "Facebook"]},
        {"brand": "  Tira  ", "impressions": 0, "clicks": 0, "leads": 0, "conversions": 0, "channels": []},
        {"impressions": 80000, "clicks": 3000, "leads": 600, "conversions": 150, "acquisition_cost": 400, "channels": ["Email"]},
    ]
    df_reg, df_cls = build_batch_campaign_rows(payloads)
    for i, p in enumerate(payloads):
        single_reg, single_cls = build_campaign_row(p)
        batch_row_reg = df_reg.iloc[[i]].reset_index(drop=True)
        batch_row_cls = df_cls.iloc[[i]].reset_index(drop=True)
        pd.testing.assert_frame_equal(batch_row_reg, single_reg, check_dtype=True)
        pd.testing.assert_frame_equal(batch_row_cls, single_cls, check_dtype=True)


def test_build_batch_channel_flags():
    payloads = [
        {"channels": ["YouTube", "Facebook"]},
        {"channels": ["Google"]},
        {"channels": None},
        {"channels": []},
        {"brand": "purplle"},  # Omitted channels defaults to ["Instagram", "Google"]
    ]
    df_reg, _ = build_batch_campaign_rows(payloads)
    for ch in CHANNELS:
        assert f"channel_{ch.lower()}" in df_reg.columns

    # Row 0: YouTube & Facebook
    assert df_reg["channel_youtube"].iloc[0] == 1
    assert df_reg["channel_facebook"].iloc[0] == 1
    assert df_reg["channel_instagram"].iloc[0] == 0

    # Row 1: Google
    assert df_reg["channel_google"].iloc[1] == 1
    assert df_reg["channel_youtube"].iloc[1] == 0

    # Row 2 (None), 3 ([]), 4 (omitted): default Instagram & Google
    for idx in (2, 3, 4):
        assert df_reg["channel_instagram"].iloc[idx] == 1
        assert df_reg["channel_google"].iloc[idx] == 1
        assert df_reg["channel_youtube"].iloc[idx] == 0


def test_build_batch_cyclical_bounds():
    months = [1, 3, 6, 9, 12]
    payloads = [{"month": m} for m in months]
    df_reg, _ = build_batch_campaign_rows(payloads)

    assert (df_reg["month_sin"] >= -1.0).all() and (df_reg["month_sin"] <= 1.0).all()
    assert (df_reg["month_cos"] >= -1.0).all() and (df_reg["month_cos"] <= 1.0).all()


def test_build_batch_zero_safe_ratios():
    payloads = [
        {"impressions": 0, "clicks": 0},
        {"clicks": 0, "conversions": 0},
        {"leads": 0, "acquisition_cost": 500.0},
        {"impressions": 0, "clicks": 0, "leads": 0, "conversions": 0, "acquisition_cost": 0},
    ]
    df_reg, _ = build_batch_campaign_rows(payloads)
    for col in ("ctr", "conversion_rate", "cpl"):
        assert not df_reg[col].isna().any()
        assert not (df_reg[col] == float("inf")).any()
        assert not (df_reg[col] == float("-inf")).any()

    assert df_reg["ctr"].iloc[0] == 0.0
    assert df_reg["conversion_rate"].iloc[1] == 0.0
    assert df_reg["cpl"].iloc[2] == 0.0


def test_build_batch_classifier_is_copy():
    df_reg, df_cls = build_batch_campaign_rows([{"brand": "purplle"}, {"brand": "nykaa"}])
    df_cls["revenue"] = [12345.0, 67890.0]
    assert "revenue" not in df_reg.columns
    assert df_cls["revenue"].tolist() == [12345.0, 67890.0]


def test_build_batch_brand_normalization():
    payloads = [
        {"brand": " Nykaa "},
        {"brand": "PURPLLE"},
        {"brand": "tira"},
    ]
    df_reg, _ = build_batch_campaign_rows(payloads)
    assert df_reg["brand"].tolist() == ["nykaa", "purplle", "tira"]


def test_build_batch_invalid_inputs_resilience():
    payloads = [
        {"impressions": "invalid_num", "clicks": "bad_clicks", "brand": None, "month": "bad_month"},
        {"impressions": 0, "clicks": 0, "leads": 0, "conversions": 0, "acquisition_cost": 0},
        {"channels": ["NonExistentChannel", 999]},
    ]
    df_reg, df_cls = build_batch_campaign_rows(payloads)
    assert len(df_reg) == 3
    assert len(df_cls) == 3

    # Coerced fallbacks
    assert df_reg["impressions"].iloc[0] == 50000.0
    assert df_reg["clicks"].iloc[0] == 4000.0
    assert df_reg["brand"].iloc[0] == "nykaa"
    assert -1.0 <= df_reg["month_sin"].iloc[0] <= 1.0

    # Zero-safe ratios
    assert df_reg["ctr"].iloc[1] == 0.0
    assert df_reg["conversion_rate"].iloc[1] == 0.0
    assert df_reg["cpl"].iloc[1] == 0.0

    # Non-existent channels are all 0
    for ch in CHANNELS:
        assert df_reg[f"channel_{ch.lower()}"].iloc[2] == 0

