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
    reg.predict.return_value = [125000.0]
    clf = MagicMock()
    clf.predict.return_value = [1]

    def _mock_load(path: str):
        if "profit" in path:
            return clf
        return reg

    with patch("api.main._load_model", side_effect=_mock_load):
        from api.main import app

        yield TestClient(app)


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

