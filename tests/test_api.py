from datetime import datetime
import json
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture
def client():
    reg = MagicMock()
    reg.predict.side_effect = lambda df: [125000.0] * len(df)
    clf = MagicMock()
    clf.predict.side_effect = lambda df: [1] * len(df)

    from api.main import app

    app.state.models = {
        "revenue_regressor": reg,
        "profit_classifier": clf,
    }
    yield TestClient(app)
    if hasattr(app.state, "models") and isinstance(app.state.models, dict):
        app.state.models.clear()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert data["revenue_model"] is True
    assert data["profit_model"] is True
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0


def test_forecast_revenue(client):
    response = client.post("/forecast_revenue", json={"brand": "nykaa"})
    assert response.status_code == 200
    assert response.json()["forecasted_revenue"] == 125000.0


def test_predict_profitability(client):
    response = client.post("/predict_profitability", json={"brand": "nykaa"})
    body = response.json()
    assert body["profitable"] is True
    assert body["status"] == "PROFITABLE"


@pytest.mark.parametrize(
    "field",
    [
        "impressions",
        "clicks",
        "leads",
        "conversions",
        "acquisition_cost",
        "engagement_score",
    ],
)
def test_negative_metric_invariants(client, field):
    response = client.post("/forecast_revenue", json={field: -1.0})
    assert response.status_code == 422
    assert field in response.text

    response_clf = client.post("/predict_profitability", json={field: -1.0})
    assert response_clf.status_code == 422
    assert field in response_clf.text


def test_clicks_exceed_impressions_invariant(client):
    response = client.post("/forecast_revenue", json={"impressions": 100, "clicks": 150})
    assert response.status_code == 422
    assert "clicks" in response.text
    assert "impressions" in response.text


def test_conversions_exceed_clicks_invariant(client):
    response = client.post(
        "/forecast_revenue",
        json={"impressions": 1000, "clicks": 50, "conversions": 80, "leads": 20},
    )
    assert response.status_code == 422
    assert "conversions" in response.text
    assert "clicks" in response.text


def test_leads_exceed_clicks_invariant(client):
    response = client.post(
        "/forecast_revenue",
        json={"impressions": 1000, "clicks": 100, "conversions": 50, "leads": 120},
    )
    assert response.status_code == 422
    assert "leads" in response.text
    assert "clicks" in response.text


@pytest.mark.parametrize("invalid_month", [0, 13, -5])
def test_month_range_validation(client, invalid_month):
    response = client.post("/forecast_revenue", json={"month": invalid_month})
    assert response.status_code == 422
    assert "month" in response.text


def test_brand_categorical_constraint(client):
    response = client.post("/forecast_revenue", json={"brand": "sephora"})
    assert response.status_code == 422
    text = response.text.lower()
    assert "nykaa" in text and "purplle" in text and "tira" in text


@pytest.mark.parametrize("brand", ["Nykaa", "PURPLLE", "tira", "  NyKaa  "])
def test_brand_case_normalization(client, brand):
    res_rev = client.post("/forecast_revenue", json={"brand": brand})
    assert res_rev.status_code == 200
    assert "forecasted_revenue" in res_rev.json()

    res_prof = client.post("/predict_profitability", json={"brand": brand})
    assert res_prof.status_code == 200
    assert "profitable" in res_prof.json()


def test_zero_input_campaign(client):
    zero_payload = {
        "impressions": 0.0,
        "clicks": 0.0,
        "leads": 0.0,
        "conversions": 0.0,
        "acquisition_cost": 0.0,
    }
    res_rev = client.post("/forecast_revenue", json=zero_payload)
    assert res_rev.status_code == 200
    assert res_rev.json()["forecasted_revenue"] == 125000.0

    res_prof = client.post("/predict_profitability", json=zero_payload)
    assert res_prof.status_code == 200
    assert res_prof.json()["status"] in ("PROFITABLE", "LOSS")


def test_campaign_input_direct_model_validation():
    from pydantic import ValidationError
    from api.main import CampaignInput

    # Valid default instantiation
    ci = CampaignInput()
    assert ci.brand == "nykaa"
    assert ci.month == 5

    # Case-insensitive brand normalization
    assert CampaignInput(brand="NyKaa").brand == "nykaa"
    assert CampaignInput(brand="PURPLLE").brand == "purplle"
    assert CampaignInput(brand="  tira  ").brand == "tira"

    # Unsupported brand rejection
    with pytest.raises(ValidationError) as exc_brand:
        CampaignInput(brand="sephora")
    assert "nykaa" in str(exc_brand.value).lower()

    # Logical cross-field invariants
    with pytest.raises(ValidationError):
        CampaignInput(impressions=100, clicks=150)
    with pytest.raises(ValidationError):
        CampaignInput(clicks=50, conversions=60)
    with pytest.raises(ValidationError):
        CampaignInput(clicks=50, leads=60)

    # Month range boundary invariants
    with pytest.raises(ValidationError):
        CampaignInput(month=0)
    with pytest.raises(ValidationError):
        CampaignInput(month=13)

    # Non-negative boundary invariants
    with pytest.raises(ValidationError):
        CampaignInput(impressions=-1.0)
    with pytest.raises(ValidationError):
        CampaignInput(clicks=-5.0)
    with pytest.raises(ValidationError):
        CampaignInput(leads=-1.0)
    with pytest.raises(ValidationError):
        CampaignInput(conversions=-1.0)
    with pytest.raises(ValidationError):
        CampaignInput(engagement_score=-0.5)
    with pytest.raises(ValidationError):
        CampaignInput(acquisition_cost=-10.0)


def test_lifespan_in_memory_models_and_zero_disk_reads():
    from api.main import app

    with TestClient(app) as test_client:
        assert hasattr(app.state, "models")
        assert "revenue_regressor" in app.state.models
        assert "profit_classifier" in app.state.models
        assert app.state.models["revenue_regressor"] is not None
        assert app.state.models["profit_classifier"] is not None

        # Verify zero disk reads during request processing
        with patch("joblib.load", side_effect=RuntimeError("Disk read attempted during inference!")):
            rev_res = test_client.post("/forecast_revenue", json={"brand": "nykaa"})
            assert rev_res.status_code == 200
            assert "forecasted_revenue" in rev_res.json()
            assert isinstance(rev_res.json()["forecasted_revenue"], float)

            prof_res = test_client.post("/predict_profitability", json={"brand": "nykaa"})
            assert prof_res.status_code == 200
            assert "forecasted_revenue" in prof_res.json()
            assert "profitable" in prof_res.json()
            assert prof_res.json()["status"] in ("PROFITABLE", "LOSS")

    # Verify shutdown cleaned up app.state.models
    assert len(app.state.models) == 0


def test_endpoints_return_503_when_models_unloaded():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        app.state.models = {}
        test_client = TestClient(app)

        res_rev = test_client.post("/forecast_revenue", json={"brand": "nykaa"})
        assert res_rev.status_code == 503
        assert "Revenue regressor model not loaded" in res_rev.json()["detail"]

        res_prof = test_client.post("/predict_profitability", json={"brand": "nykaa"})
        assert res_prof.status_code == 503
        assert "Models not loaded" in res_prof.json()["detail"]
    finally:
        app.state.models = original_models


def test_partial_models_loaded_return_503():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        # Only revenue model present
        mock_reg = MagicMock()
        mock_reg.predict.return_value = [100000.0]
        app.state.models = {"revenue_regressor": mock_reg}
        test_client = TestClient(app)

        res_rev = test_client.post("/forecast_revenue", json={"brand": "nykaa"})
        assert res_rev.status_code == 200

        res_prof = test_client.post("/predict_profitability", json={"brand": "nykaa"})
        assert res_prof.status_code == 503
        assert "Models not loaded in app.state.models" in res_prof.json()["detail"]

        # Only profit model present
        mock_clf = MagicMock()
        mock_clf.predict.return_value = [1]
        app.state.models = {"profit_classifier": mock_clf}

        res_rev2 = test_client.post("/forecast_revenue", json={"brand": "nykaa"})
        assert res_rev2.status_code == 503
        assert "Revenue regressor model not loaded" in res_rev2.json()["detail"]

        res_prof2 = test_client.post("/predict_profitability", json={"brand": "nykaa"})
        assert res_prof2.status_code == 503
    finally:
        app.state.models = original_models


def test_batch_forecast_revenue_5_items_ac_batch_01():
    from api.main import app

    items = [
        {"brand": "nykaa", "month": 1, "impressions": 10000.0, "clicks": 500.0, "leads": 100.0, "conversions": 50.0},
        {"brand": "purplle", "month": 5, "impressions": 50000.0, "clicks": 2500.0, "leads": 500.0, "conversions": 200.0},
        {"brand": "tira", "month": 8, "impressions": 30000.0, "clicks": 1200.0, "leads": 300.0, "conversions": 100.0},
        {"brand": "nykaa", "month": 12, "impressions": 80000.0, "clicks": 4000.0, "leads": 800.0, "conversions": 300.0},
        {"brand": "purplle", "month": 3, "impressions": 20000.0, "clicks": 800.0, "leads": 200.0, "conversions": 80.0},
    ]

    with TestClient(app) as test_client:
        # Test dict format {"items": [...]}
        res_dict = test_client.post("/forecast_revenue/batch", json={"items": items})
        assert res_dict.status_code == 200
        data_dict = res_dict.json()
        assert data_dict["total_items"] == 5
        assert len(data_dict["predictions"]) == 5
        assert data_dict["latency_ms"] >= 0.0

        # Test raw list format [...]
        res_list = test_client.post("/forecast_revenue/batch", json=items)
        assert res_list.status_code == 200
        data_list = res_list.json()
        assert data_list["total_items"] == 5
        assert len(data_list["predictions"]) == 5

        # Verify each prediction matches the individual /forecast_revenue endpoint
        for i, item in enumerate(items):
            single_res = test_client.post("/forecast_revenue", json=item)
            assert single_res.status_code == 200
            single_revenue = single_res.json()["forecasted_revenue"]
            assert pytest.approx(data_dict["predictions"][i], rel=1e-4) == single_revenue
            assert pytest.approx(data_list["predictions"][i], rel=1e-4) == single_revenue


def test_batch_predict_profitability_3_items_ac_batch_02():
    from api.main import app

    items = [
        {"brand": "nykaa", "month": 2, "impressions": 15000.0, "clicks": 600.0, "leads": 120.0, "conversions": 60.0},
        {"brand": "purplle", "month": 6, "impressions": 45000.0, "clicks": 2000.0, "leads": 400.0, "conversions": 150.0},
        {"brand": "tira", "month": 10, "impressions": 35000.0, "clicks": 1400.0, "leads": 350.0, "conversions": 120.0},
    ]

    with TestClient(app) as test_client:
        # Test dict format {"items": [...]}
        res = test_client.post("/predict_profitability/batch", json={"items": items})
        assert res.status_code == 200
        data = res.json()
        assert data["total_items"] == 3
        assert len(data["predictions"]) == 3
        assert data["latency_ms"] >= 0.0

        # Test raw list format [...]
        res_list = test_client.post("/predict_profitability/batch", json=items)
        assert res_list.status_code == 200
        data_list = res_list.json()
        assert data_list["total_items"] == 3
        assert len(data_list["predictions"]) == 3

        for i, pred in enumerate(data["predictions"]):
            assert "forecasted_revenue" in pred
            assert isinstance(pred["forecasted_revenue"], float)
            assert "profitable" in pred
            assert isinstance(pred["profitable"], bool)
            assert pred["status"] in ("PROFITABLE", "LOSS")

            # Verify matches individual endpoint output
            single_res = test_client.post("/predict_profitability", json=items[i])
            assert single_res.status_code == 200
            single_data = single_res.json()
            assert pytest.approx(pred["forecasted_revenue"], rel=1e-4) == single_data["forecasted_revenue"]
            assert pred["profitable"] == single_data["profitable"]
            assert pred["status"] == single_data["status"]
            assert data_list["predictions"][i]["profitable"] == single_data["profitable"]


@pytest.mark.parametrize("empty_payload", [[], {"items": []}])
def test_batch_empty_array_ac_batch_03(client, empty_payload):
    res_rev = client.post("/forecast_revenue/batch", json=empty_payload)
    assert res_rev.status_code == 422
    assert "at least 1 item" in res_rev.text.lower()

    res_prof = client.post("/predict_profitability/batch", json=empty_payload)
    assert res_prof.status_code == 422
    assert "at least 1 item" in res_prof.text.lower()


def test_batch_item_invariant_violation_index_3_ac_batch_04(client):
    items = [{"brand": "nykaa", "impressions": 50000.0, "clicks": 4000.0} for _ in range(10)]
    # Violate invariant clicks <= impressions at index 3
    items[3] = {"brand": "nykaa", "impressions": 100.0, "clicks": 150.0}

    res_rev = client.post("/forecast_revenue/batch", json={"items": items})
    assert res_rev.status_code == 422
    rev_errors = res_rev.json()["detail"]
    assert any(3 in err.get("loc", []) for err in rev_errors)
    assert "clicks" in res_rev.text and "impressions" in res_rev.text

    res_prof = client.post("/predict_profitability/batch", json={"items": items})
    assert res_prof.status_code == 422
    prof_errors = res_prof.json()["detail"]
    assert any(3 in err.get("loc", []) for err in prof_errors)
    assert "clicks" in res_prof.text and "impressions" in res_prof.text


def test_batch_oversized_payload_exceeding_500_items_ac_batch_05(client):
    oversized = [{"brand": "nykaa"} for _ in range(501)]

    res_rev = client.post("/forecast_revenue/batch", json={"items": oversized})
    assert res_rev.status_code == 422
    assert "at most 500 items" in res_rev.text.lower() or "500" in res_rev.text

    res_prof = client.post("/predict_profitability/batch", json={"items": oversized})
    assert res_prof.status_code == 422
    assert "at most 500 items" in res_prof.text.lower() or "500" in res_prof.text


def test_batch_other_invariant_violations_specify_index(client):
    items = [
        {"brand": "nykaa"},
        {"brand": "purplle", "acquisition_cost": -10.0},  # index 1: negative
        {"brand": "tira", "clicks": 100.0, "conversions": 150.0},  # index 2: conversions > clicks
        {"brand": "invalid_brand"},  # index 3: invalid brand
    ]
    res = client.post("/forecast_revenue/batch", json={"items": items})
    assert res.status_code == 422
    errors = res.json()["detail"]
    # Check that error locations point to item indices 1, 2, 3
    error_indices = {loc for err in errors for loc in err.get("loc", []) if isinstance(loc, int)}
    assert 1 in error_indices
    assert 2 in error_indices
    assert 3 in error_indices


def test_batch_invalid_input_payload_types(client):
    # Non-list items property
    res_str = client.post("/forecast_revenue/batch", json={"items": "invalid_string"})
    assert res_str.status_code == 422

    res_str_prof = client.post("/predict_profitability/batch", json={"items": "invalid_string"})
    assert res_str_prof.status_code == 422

    # List of invalid primitive items instead of objects
    res_prim = client.post("/forecast_revenue/batch", json={"items": [123, 456]})
    assert res_prim.status_code == 422

    # Batch item with invalid month value
    res_month = client.post("/forecast_revenue/batch", json={"items": [{"brand": "nykaa", "month": 13}]})
    assert res_month.status_code == 422
    assert "month" in res_month.text


def test_batch_endpoints_return_503_when_models_unloaded():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        app.state.models = {}
        test_client = TestClient(app)

        res_rev = test_client.post("/forecast_revenue/batch", json=[{"brand": "nykaa"}])
        assert res_rev.status_code == 503
        assert "Revenue regressor model not loaded" in res_rev.json()["detail"]

        res_prof = test_client.post("/predict_profitability/batch", json=[{"brand": "nykaa"}])
        assert res_prof.status_code == 503
        assert "Models not loaded" in res_prof.json()["detail"]
    finally:
        app.state.models = original_models


def test_batch_endpoints_partial_models_loaded_return_503():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        mock_reg = MagicMock()
        mock_reg.predict.side_effect = lambda df: [100000.0] * len(df)
        app.state.models = {"revenue_regressor": mock_reg}
        test_client = TestClient(app)

        res_rev = test_client.post("/forecast_revenue/batch", json=[{"brand": "nykaa"}])
        assert res_rev.status_code == 200

        res_prof = test_client.post("/predict_profitability/batch", json=[{"brand": "nykaa"}])
        assert res_prof.status_code == 503
        assert "Models not loaded in app.state.models" in res_prof.json()["detail"]
    finally:
        app.state.models = original_models


def test_batch_pydantic_schemas_direct():
    from pydantic import ValidationError
    from api.main import (
        BatchCampaignInput,
        BatchRevenueResponse,
        BatchProfitabilityResponse,
        ProfitabilityPredictionResponse,
    )

    # Valid list directly
    b_input = BatchCampaignInput.model_validate([{"brand": "nykaa"}, {"brand": "purplle"}])
    assert len(b_input.items) == 2
    assert b_input.items[0].brand == "nykaa"
    assert b_input.items[1].brand == "purplle"

    # Empty list fails min_length
    with pytest.raises(ValidationError):
        BatchCampaignInput.model_validate([])

    # Over 500 items fails max_length
    with pytest.raises(ValidationError):
        BatchCampaignInput.model_validate([{"brand": "nykaa"} for _ in range(501)])

    # Response schemas
    rev_resp = BatchRevenueResponse(predictions=[100.0, 200.0], total_items=2, latency_ms=12.5)
    assert rev_resp.total_items == 2
    assert rev_resp.predictions == [100.0, 200.0]

    prof_resp = BatchProfitabilityResponse(
        predictions=[
            ProfitabilityPredictionResponse(forecasted_revenue=100.0, profitable=True, status="PROFITABLE")
        ],
        total_items=1,
        latency_ms=8.2,
    )
    assert prof_resp.total_items == 1
    assert prof_resp.predictions[0].profitable is True
    assert prof_resp.predictions[0].status == "PROFITABLE"


def test_app_ui_serving_and_markup(client):
    res = client.get("/app")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html_text = res.text
    assert "backdrop-filter: blur(16px)" in html_text
    assert "-webkit-backdrop-filter: blur(16px)" in html_text
    assert "Forecasted Revenue" in html_text
    assert "₹" in html_text
    assert "Nykaa" in html_text
    assert "Purplle" in html_text
    assert "Tira" in html_text
    assert "📈 Forecasted Status: PROFITABLE CAMPAIGN" in html_text
    assert "📉 Forecasted Status: NET OPERATIONAL LOSS" in html_text
    assert "Run Campaign Forecast" in html_text


def test_app_ui_trailing_slash_route(client):
    res = client.get("/app/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Forecasted Revenue" in res.text


def test_app_ui_root_redirect_307(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 307
    assert res.headers["location"] == "/app"


def test_app_ui_root_redirect_follow(client):
    res = client.get("/", follow_redirects=True)
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Forecasted Revenue" in res.text


def test_app_ui_missing_file_returns_404(client):
    with patch("api.main.INDEX_HTML_PATH", "/non/existent/path/index.html"):
        res = client.get("/app")
        assert res.status_code == 404
        assert "Single-page application not found" in res.json()["detail"]


def test_app_ui_ac_ui_01_glassmorphic_styling(client):
    res = client.get("/app")
    assert res.status_code == 200
    html = res.text
    # Frosted-glass containers
    assert "backdrop-filter: blur(16px)" in html
    assert "-webkit-backdrop-filter: blur(16px)" in html
    # Semi-transparent backgrounds
    assert "rgba(17, 24, 39" in html or "rgba(255, 255, 255, 0.04)" in html
    # Radial glowing gradients
    assert "radial-gradient" in html
    # Rounded corner borders >= 12px
    assert "border-radius: 12px" in html or "border-radius: 16px" in html or "border-radius: 18px" in html


def test_app_ui_ac_ui_02_async_fetch_and_inr_formatting(client):
    res = client.get("/app")
    assert res.status_code == 200
    html = res.text
    # Asynchronous fetch calls
    assert "fetch('/forecast_revenue'" in html
    assert "fetch('/predict_profitability'" in html
    # INR currency formatting and KPI elements
    assert "currency: 'INR'" in html
    assert "en-IN" in html
    assert "₹" in html
    assert 'id="res-revenue"' in html
    assert "Run Campaign Forecast" in html


def test_app_ui_ac_ui_03_profitable_emerald_badge(client):
    res = client.get("/app")
    assert res.status_code == 200
    html = res.text
    assert "📈 Forecasted Status: PROFITABLE CAMPAIGN" in html
    assert ".profit-badge.emerald-glass" in html
    assert "emerald-glass" in html


def test_app_ui_ac_ui_04_unprofitable_ruby_badge(client):
    res = client.get("/app")
    assert res.status_code == 200
    html = res.text
    assert "📉 Forecasted Status: NET OPERATIONAL LOSS" in html
    assert ".profit-badge.ruby-glass" in html
    assert "ruby-glass" in html


def test_app_ui_form_elements_and_brands(client):
    res = client.get("/app")
    assert res.status_code == 200
    html = res.text
    # Brand selection buttons
    for brand in ["Nykaa", "Purplle", "Tira"]:
        assert brand in html
    # Form input fields in 3-column layout
    for field_id in [
        "campaign_type",
        "target_audience",
        "customer_segment",
        "language",
        "month",
        "impressions",
        "clicks",
        "leads",
        "conversions",
        "engagement_score",
        "acquisition_cost",
    ]:
        assert f'id="{field_id}"' in html
    # Multi-channel checkboxes
    for channel in ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"]:
        assert channel in html


def test_app_ui_client_side_validation_logic(client):
    res = client.get("/app")
    assert res.status_code == 200
    html = res.text
    # Validation function and invariants
    assert "validateInvariants" in html
    assert "clicks > impressions" in html
    assert "conversions > clicks" in html
    assert "leads > clicks" in html
    assert "month < 1 || month > 12" in html
    assert "channels.length === 0" in html


def test_app_ui_service_health_pill(client):
    res = client.get("/app")
    assert res.status_code == 200
    html = res.text
    # Service health checking
    assert "fetch('/health'" in html
    assert "status-pill" in html
    assert "status-dot" in html
    assert "status-text" in html


def test_app_ui_invalid_methods(client):
    # GET and HEAD are permitted; other methods return 405 Method Not Allowed
    res_post = client.post("/app")
    assert res_post.status_code == 405
    res_put = client.put("/app")
    assert res_put.status_code == 405
    res_delete = client.delete("/app")
    assert res_delete.status_code == 405


def test_health_uptime_increases(client):
    import time

    res1 = client.get("/health")
    assert res1.status_code == 200
    time.sleep(0.02)
    res2 = client.get("/health")
    assert res2.status_code == 200
    assert res2.json()["uptime_seconds"] >= res1.json()["uptime_seconds"]


def test_health_unloaded_models():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        app.state.models = {}
        test_client = TestClient(app)
        res = test_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert data["revenue_model"] is False
        assert data["profit_model"] is False
        assert data["uptime_seconds"] >= 0.0
    finally:
        app.state.models = original_models


def test_health_response_schema_direct():
    from api.main import HealthResponse

    hr = HealthResponse(
        status="ok",
        version="1.0.0",
        revenue_model=True,
        profit_model=True,
        uptime_seconds=12.34,
    )
    assert hr.status == "ok"
    assert hr.version == "1.0.0"
    assert hr.revenue_model is True
    assert hr.profit_model is True
    assert hr.uptime_seconds == 12.34


def test_ready_success_when_models_loaded_ac_probe_01(client):
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["models_loaded"] is True
    assert "detail" not in data or data["detail"] is None


def test_ready_failure_when_models_unloaded_ac_probe_02():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        app.state.models = {}
        test_client = TestClient(app)
        res = test_client.get("/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "unready"
        assert data["models_loaded"] is False
        assert "revenue_regressor" in data["detail"]
        assert "profit_classifier" in data["detail"]
    finally:
        app.state.models = original_models


def test_ready_failure_when_partial_models_loaded():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        mock_reg = MagicMock()
        app.state.models = {"revenue_regressor": mock_reg}
        test_client = TestClient(app)

        res = test_client.get("/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "unready"
        assert data["models_loaded"] is False
        assert "profit_classifier" in data["detail"]

        mock_clf = MagicMock()
        app.state.models = {"profit_classifier": mock_clf}
        res2 = test_client.get("/ready")
        assert res2.status_code == 503
        data2 = res2.json()
        assert data2["status"] == "unready"
        assert data2["models_loaded"] is False
        assert "revenue_regressor" in data2["detail"]
    finally:
        app.state.models = original_models


def test_ready_and_health_with_lifespan():
    from api.main import app

    with TestClient(app) as test_client:
        res_ready = test_client.get("/ready")
        assert res_ready.status_code == 200
        assert res_ready.json() == {"status": "ready", "models_loaded": True}

        res_health = test_client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["version"] == "1.0.0"
        assert health_data["revenue_model"] is True
        assert health_data["profit_model"] is True
        assert health_data["uptime_seconds"] >= 0.0


def test_readiness_response_schema_direct():
    from api.main import ReadinessResponse

    r_ready = ReadinessResponse(status="ready", models_loaded=True)
    assert r_ready.status == "ready"
    assert r_ready.models_loaded is True
    assert r_ready.detail is None

    r_unready = ReadinessResponse(
        status="unready",
        models_loaded=False,
        detail="Missing revenue_regressor",
    )
    assert r_unready.status == "unready"
    assert r_unready.models_loaded is False
    assert r_unready.detail == "Missing revenue_regressor"


def test_probes_invalid_http_methods(client):
    # Probes only accept GET/HEAD; any mutating method should return 405 Method Not Allowed
    for method in ["post", "put", "delete", "patch"]:
        res_health = client.request(method, "/health")
        assert res_health.status_code == 405, f"Expected 405 for {method.upper()} /health, got {res_health.status_code}"

        res_ready = client.request(method, "/ready")
        assert res_ready.status_code == 405, f"Expected 405 for {method.upper()} /ready, got {res_ready.status_code}"


def test_ready_and_health_with_corrupt_or_none_models_state():
    from api.main import app

    original_models = getattr(app.state, "models", {})
    try:
        # Test app.state.models set to None
        app.state.models = None
        test_client = TestClient(app)

        res_health = test_client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["revenue_model"] is False
        assert health_data["profit_model"] is False

        res_ready = test_client.get("/ready")
        assert res_ready.status_code == 503
        ready_data = res_ready.json()
        assert ready_data["status"] == "unready"
        assert ready_data["models_loaded"] is False

        # Test app.state.models with explicit None values
        app.state.models = {"revenue_regressor": None, "profit_classifier": None}
        res_ready_none = test_client.get("/ready")
        assert res_ready_none.status_code == 503
        assert res_ready_none.json()["models_loaded"] is False
    finally:
        app.state.models = original_models


def test_readiness_response_schema_invalid_status_rejection():
    from pydantic import ValidationError
    from api.main import ReadinessResponse

    # Invalid status not in Literal["ready", "unready"]
    with pytest.raises(ValidationError):
        ReadinessResponse(status="degraded", models_loaded=True)

    with pytest.raises(ValidationError):
        ReadinessResponse(status="ok", models_loaded=True)


def test_health_response_schema_missing_fields_rejection():
    from pydantic import ValidationError
    from api.main import HealthResponse

    # Missing mandatory boolean and uptime fields
    with pytest.raises(ValidationError):
        HealthResponse(status="ok")

    with pytest.raises(ValidationError):
        HealthResponse(status="ok", version="1.0.0", revenue_model=True)


# ============================================================================
# M3-TASK-02: Structured JSON Logging & Latency Middleware Tests (AC-OBS-01)
# ============================================================================


def test_structured_logging_middleware_records_valid_json(client, caplog):
    """AC-OBS-01: Verify structured JSON log entry contains timestamp, method, path, status_code, latency_ms."""
    with caplog.at_level(logging.INFO, logger="api.main"):
        caplog.clear()
        response = client.get("/health")
        assert response.status_code == 200

        # Find matching log record from api.main
        api_records = [r for r in caplog.records if r.name == "api.main"]
        assert len(api_records) >= 1

        record = api_records[-1]
        log_data = json.loads(record.message)

        assert "timestamp" in log_data
        assert "method" in log_data
        assert "path" in log_data
        assert "status_code" in log_data
        assert "latency_ms" in log_data

        assert log_data["method"] == "GET"
        assert log_data["path"] == "/health"
        assert log_data["status_code"] == 200
        assert isinstance(log_data["latency_ms"], (int, float))
        assert log_data["latency_ms"] >= 0.0

        # Verify timestamp is valid ISO 8601 format
        parsed_time = datetime.fromisoformat(log_data["timestamp"])
        assert parsed_time is not None


def test_structured_logging_middleware_does_not_log_sensitive_payloads(client, caplog):
    """Verify log records never include sensitive campaign inputs, spend, or request bodies."""
    sensitive_payload = {
        "brand": "nykaa",
        "campaign_type": "Paid Ads",
        "target_audience": "Youth",
        "language": "English",
        "customer_segment": "Premium Shoppers",
        "month": 6,
        "impressions": 99999.0,
        "clicks": 8888.0,
        "leads": 2222.0,
        "conversions": 1111.0,
        "engagement_score": 25.0,
        "acquisition_cost": 499.0,
        "channels": ["Instagram", "Google"],
    }

    with caplog.at_level(logging.INFO, logger="api.main"):
        caplog.clear()
        res = client.post("/forecast_revenue", json=sensitive_payload)
        assert res.status_code == 200

        api_records = [r for r in caplog.records if r.name == "api.main"]
        assert len(api_records) >= 1

        record = api_records[-1]
        log_data = json.loads(record.message)

        # Expected keys only
        assert set(log_data.keys()) == {
            "timestamp",
            "method",
            "path",
            "status_code",
            "latency_ms",
        }
        assert log_data["method"] == "POST"
        assert log_data["path"] == "/forecast_revenue"
        assert log_data["status_code"] == 200

        # Raw log message must not leak any campaign inputs or numbers
        raw_msg = record.message
        assert "99999" not in raw_msg
        assert "8888" not in raw_msg
        assert "2222" not in raw_msg
        assert "1111" not in raw_msg
        assert "499" not in raw_msg
        assert "impressions" not in raw_msg
        assert "clicks" not in raw_msg
        assert "acquisition_cost" not in raw_msg
        assert "Premium Shoppers" not in raw_msg


def test_structured_logging_middleware_captures_client_validation_error_422(client, caplog):
    """Verify middleware records 422 Unprocessable Entity on invariant violation without leaking inputs."""
    with caplog.at_level(logging.INFO, logger="api.main"):
        caplog.clear()
        res = client.post(
            "/forecast_revenue",
            json={"impressions": 100.0, "clicks": 200.0},
        )
        assert res.status_code == 422

        api_records = [r for r in caplog.records if r.name == "api.main"]
        assert len(api_records) >= 1

        record = api_records[-1]
        log_data = json.loads(record.message)

        assert log_data["method"] == "POST"
        assert log_data["path"] == "/forecast_revenue"
        assert log_data["status_code"] == 422
        assert isinstance(log_data["latency_ms"], (int, float))
        assert log_data["latency_ms"] >= 0.0


def test_structured_logging_middleware_batch_endpoints(client, caplog):
    """Verify batch inference endpoints emit valid structured JSON logs."""
    batch_payload = [
        {"brand": "nykaa", "month": 5, "impressions": 50000.0, "clicks": 4000.0},
        {"brand": "purplle", "month": 6, "impressions": 60000.0, "clicks": 5000.0},
    ]

    with caplog.at_level(logging.INFO, logger="api.main"):
        caplog.clear()
        res_rev = client.post("/forecast_revenue/batch", json=batch_payload)
        assert res_rev.status_code == 200

        rev_record = [r for r in caplog.records if r.name == "api.main"][-1]
        rev_log = json.loads(rev_record.message)
        assert rev_log["method"] == "POST"
        assert rev_log["path"] == "/forecast_revenue/batch"
        assert rev_log["status_code"] == 200
        assert rev_log["latency_ms"] >= 0.0

        caplog.clear()
        res_prof = client.post("/predict_profitability/batch", json=batch_payload)
        assert res_prof.status_code == 200

        prof_record = [r for r in caplog.records if r.name == "api.main"][-1]
        prof_log = json.loads(prof_record.message)
        assert prof_log["method"] == "POST"
        assert prof_log["path"] == "/predict_profitability/batch"
        assert prof_log["status_code"] == 200
        assert prof_log["latency_ms"] >= 0.0


def test_structured_logging_middleware_probes_and_redirect(client, caplog):
    """Verify middleware intercepts /ready, /, and /app accurately."""
    with caplog.at_level(logging.INFO, logger="api.main"):
        # Test /ready 200
        caplog.clear()
        res_ready = client.get("/ready")
        assert res_ready.status_code == 200
        ready_log = json.loads([r for r in caplog.records if r.name == "api.main"][-1].message)
        assert ready_log["method"] == "GET"
        assert ready_log["path"] == "/ready"
        assert ready_log["status_code"] == 200

        # Test redirect / -> /app (307)
        caplog.clear()
        res_redir = client.get("/", follow_redirects=False)
        assert res_redir.status_code == 307
        redir_log = json.loads([r for r in caplog.records if r.name == "api.main"][-1].message)
        assert redir_log["method"] == "GET"
        assert redir_log["path"] == "/"
        assert redir_log["status_code"] == 307

        # Test /app (200)
        caplog.clear()
        res_app = client.get("/app")
        assert res_app.status_code == 200
        app_log = json.loads([r for r in caplog.records if r.name == "api.main"][-1].message)
        assert app_log["method"] == "GET"
        assert app_log["path"] == "/app"
        assert app_log["status_code"] == 200


def test_structured_logging_middleware_readiness_503(client, caplog):
    """Verify middleware records 503 status code when service is unready."""
    from api.main import app

    saved_models = app.state.models
    try:
        app.state.models = {}
        with caplog.at_level(logging.INFO, logger="api.main"):
            caplog.clear()
            res = client.get("/ready")
            assert res.status_code == 503

            records = [r for r in caplog.records if r.name == "api.main"]
            assert len(records) >= 1
            log_data = json.loads(records[-1].message)
            assert log_data["method"] == "GET"
            assert log_data["path"] == "/ready"
            assert log_data["status_code"] == 503
            assert log_data["latency_ms"] >= 0.0
    finally:
        app.state.models = saved_models


def test_structured_logging_middleware_unhandled_exception(caplog):
    """Verify middleware logs status 500 and re-raises on unhandled endpoint exception."""
    from api.main import app

    # Temporarily add a failing route to test unhandled exception handling
    @app.get("/test_crash_route")
    def crash_route():
        raise RuntimeError("simulated server crash")

    try:
        test_client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.INFO, logger="api.main"):
            caplog.clear()
            res = test_client.get("/test_crash_route")
            assert res.status_code == 500

            records = [r for r in caplog.records if r.name == "api.main"]
            assert len(records) >= 1
            log_data = json.loads(records[-1].message)
            assert log_data["method"] == "GET"
            assert log_data["path"] == "/test_crash_route"
            assert log_data["status_code"] == 500
            assert log_data["latency_ms"] >= 0.0
    finally:
        # Clean up route from app
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/test_crash_route"]


def test_structured_logging_middleware_captures_404_not_found(client, caplog):
    """Verify middleware intercepts non-existent route and logs 404 status code."""
    with caplog.at_level(logging.INFO, logger="api.main"):
        caplog.clear()
        res = client.get("/non_existent_endpoint")
        assert res.status_code == 404

        records = [r for r in caplog.records if r.name == "api.main"]
        assert len(records) >= 1
        log_data = json.loads(records[-1].message)
        assert log_data["method"] == "GET"
        assert log_data["path"] == "/non_existent_endpoint"
        assert log_data["status_code"] == 404
        assert isinstance(log_data["latency_ms"], (int, float))
        assert log_data["latency_ms"] >= 0.0


def test_structured_logging_middleware_malformed_json_body(client, caplog):
    """Verify middleware logs 422 on completely malformed JSON payload."""
    with caplog.at_level(logging.INFO, logger="api.main"):
        caplog.clear()
        res = client.post(
            "/forecast_revenue",
            content=b"not valid json {{{",
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 422

        records = [r for r in caplog.records if r.name == "api.main"]
        assert len(records) >= 1
        log_data = json.loads(records[-1].message)
        assert log_data["method"] == "POST"
        assert log_data["path"] == "/forecast_revenue"
        assert log_data["status_code"] == 422
        assert isinstance(log_data["latency_ms"], (int, float))
        assert log_data["latency_ms"] >= 0.0








