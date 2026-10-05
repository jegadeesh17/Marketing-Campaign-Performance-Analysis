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
    assert client.get("/health").status_code == 200


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


