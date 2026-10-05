"""FastAPI for marketing campaign forecasting."""

from __future__ import annotations

import os
import sys

from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import build_campaign_row

app = FastAPI(title="Marketing Campaign Intelligence API", version="1.0.0")

BrandType = Literal["nykaa", "purplle", "tira"]
CampaignType = Literal["Social Media", "Paid Ads", "Influencer", "Email", "SEO"]
TargetAudienceType = Literal["College Students", "Tier 2 City Customers", "Youth", "Working Women"]
CustomerSegmentType = Literal["College Students", "Premium Shoppers", "Working Women", "Tier 2 City Customers"]
LanguageType = Literal["English", "Hindi", "Tamil", "Bengali"]


class CampaignInput(BaseModel):
    brand: str = Field(default="nykaa", description="Target e-commerce brand (case-insensitive)")
    campaign_type: CampaignType = Field(default="Paid Ads", description="Marketing campaign strategy")
    target_audience: TargetAudienceType = Field(default="Youth", description="Demographic audience target")
    language: LanguageType = Field(default="English", description="Creative language context")
    customer_segment: CustomerSegmentType = Field(default="Premium Shoppers", description="Customer purchasing tier")
    month: int = Field(default=5, ge=1, le=12, description="Execution calendar month (1-12)")
    impressions: float = Field(default=50000.0, ge=0.0, description="Expected ad impressions")
    clicks: float = Field(default=4000.0, ge=0.0, description="Expected click volume")
    leads: float = Field(default=1500.0, ge=0.0, description="Projected inbound leads")
    conversions: float = Field(default=500.0, ge=0.0, description="Target customer conversions")
    engagement_score: float = Field(default=15.0, ge=0.0, description="Target engagement score (0-30)")
    acquisition_cost: float = Field(default=250.0, ge=0.0, description="Cost Per Acquisition in INR")
    channels: list[str] = Field(
        default_factory=lambda: ["Instagram", "Google"],
        description="Delivery channels (YouTube, Instagram, Google, WhatsApp, Email, Facebook)",
    )

    @field_validator("brand", mode="before")
    @classmethod
    def normalize_brand(cls, v: str) -> str:
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized not in {"nykaa", "purplle", "tira"}:
                raise ValueError(f"Invalid brand '{v}'. Supported brands are: ['nykaa', 'purplle', 'tira']")
            return normalized
        raise ValueError("Brand must be a valid string")

    @model_validator(mode="after")
    def validate_logical_invariants(self) -> CampaignInput:
        if self.clicks > self.impressions:
            raise ValueError(
                f"Logical invariant violated: clicks ({self.clicks}) cannot exceed impressions ({self.impressions})"
            )
        if self.conversions > self.clicks:
            raise ValueError(
                f"Logical invariant violated: conversions ({self.conversions}) cannot exceed clicks ({self.clicks})"
            )
        if self.leads > self.clicks:
            raise ValueError(
                f"Logical invariant violated: leads ({self.leads}) cannot exceed clicks ({self.clicks})"
            )
        return self


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
