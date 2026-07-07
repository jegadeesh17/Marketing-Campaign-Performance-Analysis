"""FastAPI for marketing campaign forecasting."""

from __future__ import annotations

import os
import sys

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import build_campaign_row

app = FastAPI(title="Marketing Campaign Intelligence API", version="1.0.0")


class CampaignInput(BaseModel):
    brand: str = "nykaa"
    campaign_type: str = "Paid Ads"
    target_audience: str = "Youth"
    language: str = "English"
    customer_segment: str = "Premium Shoppers"
    month: int = Field(default=5, ge=1, le=12)
    impressions: float = 50000
    clicks: float = 4000
    leads: float = 1500
    conversions: float = 500
    engagement_score: float = 15.0
    acquisition_cost: float = 250.0
    channels: list[str] = Field(default_factory=lambda: ["Instagram", "Google"])


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model: {path}. Run python src/train_models.py")
    return joblib.load(path)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "revenue_model": os.path.exists(os.path.join(ROOT, "models", "revenue_regressor.joblib")),
        "profit_model": os.path.exists(os.path.join(ROOT, "models", "profit_classifier.joblib")),
    }


@app.post("/forecast_revenue")
def forecast_revenue(campaign: CampaignInput) -> dict:
    df_reg, _ = build_campaign_row(campaign.model_dump())
    try:
        model = _load_model(os.path.join(ROOT, "models", "revenue_regressor.joblib"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    revenue = float(model.predict(df_reg)[0])
    return {"forecasted_revenue": revenue}


@app.post("/predict_profitability")
def predict_profitability(campaign: CampaignInput) -> dict:
    df_reg, df_cls = build_campaign_row(campaign.model_dump())
    try:
        reg = _load_model(os.path.join(ROOT, "models", "revenue_regressor.joblib"))
        clf = _load_model(os.path.join(ROOT, "models", "profit_classifier.joblib"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    revenue = float(reg.predict(df_reg)[0])
    df_cls["revenue"] = revenue
    profit_flag = int(clf.predict(df_cls)[0])
    return {
        "forecasted_revenue": revenue,
        "profitable": profit_flag == 1,
        "status": "PROFITABLE" if profit_flag == 1 else "LOSS",
    }
