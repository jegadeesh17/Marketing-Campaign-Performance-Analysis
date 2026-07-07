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
