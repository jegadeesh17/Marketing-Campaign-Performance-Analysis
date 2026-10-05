"""FastAPI for marketing campaign forecasting."""

from __future__ import annotations

import os
import sys
import time

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import get_settings
from src.inference import build_batch_campaign_rows, build_campaign_row

# Ensure backwards compatibility for models pickled in scikit-learn 1.6
try:
    import sklearn.compose._column_transformer as _ct

    if not hasattr(_ct, "_RemainderColsList"):

        class _RemainderColsList(list):
            def __setstate__(self, state):
                if isinstance(state, dict):
                    if "data" in state:
                        self.extend(state["data"])
                    self.__dict__.update(state)
                elif isinstance(state, (list, tuple)):
                    self.extend(state)

        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load XGBoost models into app.state.models on startup; clear cache on shutdown."""
    settings = get_settings()
    models: dict[str, Any] = {}
    model_dir = (
        settings.model_dir
        if os.path.isabs(settings.model_dir)
        else os.path.join(ROOT, settings.model_dir)
    )
    revenue_model_path = os.path.join(model_dir, "revenue_regressor.joblib")
    profit_model_path = os.path.join(model_dir, "profit_classifier.joblib")

    if os.path.exists(revenue_model_path):
        models["revenue_regressor"] = joblib.load(revenue_model_path)
    if os.path.exists(profit_model_path):
        models["profit_classifier"] = joblib.load(profit_model_path)

    app.state.models = models
    yield
    if hasattr(app.state, "models") and isinstance(app.state.models, dict):
        app.state.models.clear()


app = FastAPI(
    title="Marketing Campaign Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.models = {}
app.state.settings = get_settings()

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


class BatchCampaignInput(BaseModel):
    items: list[CampaignInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of campaign records for bulk evaluation (1 to 500 items)",
    )

    @model_validator(mode="before")
    @classmethod
    def check_list_or_dict(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"items": data}
        return data


class RevenuePredictionResponse(BaseModel):
    forecasted_revenue: float = Field(..., description="Projected gross revenue in INR")


class BatchRevenueResponse(BaseModel):
    predictions: list[float] = Field(..., description="Ordered list of predicted revenues in INR")
    total_items: int = Field(..., description="Total records evaluated in the batch")
    latency_ms: float = Field(..., description="Batch inference processing latency in milliseconds")


class ProfitabilityPredictionResponse(BaseModel):
    forecasted_revenue: float = Field(..., description="Projected gross revenue in INR")
    profitable: bool = Field(..., description="Binary classification flag indicating profitability")
    status: Literal["PROFITABLE", "LOSS"] = Field(..., description="Human-readable business outcome")


class BatchProfitabilityResponse(BaseModel):
    predictions: list[ProfitabilityPredictionResponse] = Field(
        ..., description="Ordered list of profitability predictions"
    )
    total_items: int = Field(..., description="Total records evaluated in the batch")
    latency_ms: float = Field(..., description="Batch inference processing latency in milliseconds")


def _load_model(path: str):
    if not os.path.exists(path):
        settings = get_settings()
        model_dir = (
            settings.model_dir
            if os.path.isabs(settings.model_dir)
            else os.path.join(ROOT, settings.model_dir)
        )
        alt_path = os.path.join(model_dir, os.path.basename(path))
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Missing model: {path}. Run python src/train_models.py")
    return joblib.load(path)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    model_dir = (
        settings.model_dir
        if os.path.isabs(settings.model_dir)
        else os.path.join(ROOT, settings.model_dir)
    )
    models = getattr(app.state, "models", {})
    revenue_loaded = bool(models.get("revenue_regressor") or models.get("revenue"))
    profit_loaded = bool(models.get("profit_classifier") or models.get("profit"))
    return {
        "status": "ok",
        "revenue_model": revenue_loaded or os.path.exists(os.path.join(model_dir, "revenue_regressor.joblib")),
        "profit_model": profit_loaded or os.path.exists(os.path.join(model_dir, "profit_classifier.joblib")),
    }


@app.post("/forecast_revenue", response_model=RevenuePredictionResponse)
def forecast_revenue(campaign: CampaignInput) -> dict:
    models = getattr(app.state, "models", {})
    model = models.get("revenue_regressor") or models.get("revenue")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Revenue regressor model not loaded in app.state.models",
        )
    df_reg, _ = build_campaign_row(campaign.model_dump())
    revenue = float(model.predict(df_reg)[0])
    return {"forecasted_revenue": revenue}


@app.post("/predict_profitability", response_model=ProfitabilityPredictionResponse)
def predict_profitability(campaign: CampaignInput) -> dict:
    models = getattr(app.state, "models", {})
    reg = models.get("revenue_regressor") or models.get("revenue")
    clf = models.get("profit_classifier") or models.get("profit")
    if reg is None or clf is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded in app.state.models",
        )
    df_reg, df_cls = build_campaign_row(campaign.model_dump())
    revenue = float(reg.predict(df_reg)[0])
    df_cls["revenue"] = revenue
    profit_flag = int(clf.predict(df_cls)[0])
    return {
        "forecasted_revenue": revenue,
        "profitable": profit_flag == 1,
        "status": "PROFITABLE" if profit_flag == 1 else "LOSS",
    }


@app.post("/forecast_revenue/batch", response_model=BatchRevenueResponse)
def forecast_revenue_batch(batch: BatchCampaignInput) -> BatchRevenueResponse:
    start_time = time.perf_counter()
    models = getattr(app.state, "models", {})
    model = models.get("revenue_regressor") or models.get("revenue")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Revenue regressor model not loaded in app.state.models",
        )
    payloads = [item.model_dump() for item in batch.items]
    df_reg, _ = build_batch_campaign_rows(payloads)
    predictions = [float(val) for val in model.predict(df_reg)]
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    return BatchRevenueResponse(
        predictions=predictions,
        total_items=len(predictions),
        latency_ms=latency_ms,
    )


@app.post("/predict_profitability/batch", response_model=BatchProfitabilityResponse)
def predict_profitability_batch(batch: BatchCampaignInput) -> BatchProfitabilityResponse:
    start_time = time.perf_counter()
    models = getattr(app.state, "models", {})
    reg = models.get("revenue_regressor") or models.get("revenue")
    clf = models.get("profit_classifier") or models.get("profit")
    if reg is None or clf is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded in app.state.models",
        )
    payloads = [item.model_dump() for item in batch.items]
    df_reg, df_cls = build_batch_campaign_rows(payloads)
    pred_revenues = [float(val) for val in reg.predict(df_reg)]
    df_cls["revenue"] = pred_revenues
    pred_flags = clf.predict(df_cls)

    predictions: list[ProfitabilityPredictionResponse] = []
    for rev, flag in zip(pred_revenues, pred_flags):
        is_profitable = int(flag) == 1
        predictions.append(
            ProfitabilityPredictionResponse(
                forecasted_revenue=rev,
                profitable=is_profitable,
                status="PROFITABLE" if is_profitable else "LOSS",
            )
        )
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    return BatchProfitabilityResponse(
        predictions=predictions,
        total_items=len(predictions),
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    effective_port = settings.port if settings.port != 8000 else settings.api_port
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=effective_port,
        reload=(settings.app_env == "development"),
    )

