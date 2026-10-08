# Marketing Campaign Intelligence — 5-Minute Showcase Demo Script

A complete 5-minute technical demonstration script for stakeholders, growth leads, and engineering interviews. Demonstrates operational readiness, lifespan in-memory model serving, high-throughput batch inference, boundary invariant safety, and the unified dark-mode glassmorphic single-page web app.

---

## 0. Prerequisites & Environment Setup (30 sec)

Clone the repository and install runtime dependencies:

```bash
pip install -r requirements.txt
```

Verify that pre-trained XGBoost model artifacts exist in `models/` (or re-train if needed):

```bash
python src/train_models.py
```

### Option A: Launch Standalone Uvicorn Microservice (Single Port 8000)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Option B: Launch Unified Single-Container Docker Topology

```bash
docker compose up --build
```

> **Architecture Note:** As defined in [ADR-0007](DECISIONS.md#adr-0007-elimination-of-streamlit-in-favor-of-unified-fastapi-glassmorphic-web-application), the platform runs as a single unified Uvicorn ASGI microservice listening on port 8000. All Streamlit processes have been eliminated, reducing memory overhead by >50% and simplifying Cloud Run deployment.

---

## 1. Operational Probes & Model Lifespan Inspection (45 sec)

Demonstrate production Kubernetes/Cloud Run lifecycle management via separated liveness and readiness probes ([ADR-0005](DECISIONS.md#adr-0005-operational-liveness-and-readiness-probes-separation)).

### 1.1 Liveness Probe (`GET /health`)

Query microservice uptime, semantic version, and model flags:

```bash
curl http://127.0.0.1:8000/health
```

**Expected Response (`HTTP 200`):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "revenue_model": true,
  "profit_model": true,
  "uptime_seconds": 14.82
}
```

### 1.2 Readiness Probe (`GET /ready`)

Confirm that XGBoost regression and classification pipelines are pre-warmed in `app.state.models` ([ADR-0002](DECISIONS.md#adr-0002-in-memory-model-lifecycle-management-via-fastapi-lifespan)):

```bash
curl http://127.0.0.1:8000/ready
```

**Expected Response (`HTTP 200`):**
```json
{
  "status": "ready",
  "models_loaded": true
}
```

---

## 2. Dark-Mode Glassmorphic Web App Walkthrough (1 min 30 sec)

Open your browser and navigate to:

```text
http://127.0.0.1:8000/app
```

*(Note: Visiting `http://127.0.0.1:8000/` automatically returns an HTTP 307 temporary redirect to `/app`)*.

### Key Visual & Interactive Elements to Showcase:
1. **Glassmorphic Aesthetic:** Point out the frosted-glass containers (`backdrop-filter: blur(16px)`), radial glowing accents, and live service health indicator pill ("● Service Online").
2. **Multi-Brand Switcher:** Click through the branded selection chips: **Nykaa** (magenta glow), **Purplle** (violet glow), and **Tira** (amber glow).
3. **Execution Parameter Controls:** Adjust the 3-column input configuration:
   - Campaign Strategy: `Paid Ads`
   - Target Audience: `Youth`
   - Customer Segment: `Premium Shoppers`
   - Calendar Month slider: `Month 5`
   - Delivery Channels: Check `Instagram` and `Google`
4. **Instant Client-Side Boundary Validation:** Try entering impressions lower than clicks; note the immediate real-time visual warning before submission.
5. **Campaign Simulation Execution:** Click **"🚀 Run Campaign Forecast"**:
   - The app dispatches asynchronous client-side `fetch()` requests to `POST /forecast_revenue` and `POST /predict_profitability`.
   - Displays **Forecasted Revenue in INR** (`₹275,122.22`).
   - Displays **Profitability Status**: Emerald glass badge (`📈 PROFITABLE CAMPAIGN`) or Ruby glass badge (`📉 NET OPERATIONAL LOSS`).
   - Real-time efficiency scorecards: CTR %, Conversion Rate %, and Cost Per Lead (CPL).

---

## 3. High-Performance REST API Inference (1 min 15 sec)

Demonstrate single-record and high-throughput batch prediction endpoints for programmatic bidding integration.

### 3.1 Single Campaign Revenue Forecasting (`POST /forecast_revenue`)

```bash
curl -X POST http://127.0.0.1:8000/forecast_revenue \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "nykaa",
    "campaign_type": "Paid Ads",
    "target_audience": "Youth",
    "language": "English",
    "customer_segment": "Premium Shoppers",
    "month": 5,
    "impressions": 50000.0,
    "clicks": 4000.0,
    "leads": 1500.0,
    "conversions": 500.0,
    "engagement_score": 15.0,
    "acquisition_cost": 250.0,
    "channels": ["Instagram", "Google"]
  }'
```

**Expected Response (`HTTP 200`):**
```json
{
  "forecasted_revenue": 275122.22
}
```

### 3.2 Single Campaign Profitability Classification (`POST /predict_profitability`)

```bash
curl -X POST http://127.0.0.1:8000/predict_profitability \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "purplle",
    "campaign_type": "Social Media",
    "target_audience": "College Students",
    "language": "Hindi",
    "customer_segment": "College Students",
    "month": 6,
    "impressions": 40000.0,
    "clicks": 3000.0,
    "leads": 1000.0,
    "conversions": 300.0,
    "engagement_score": 18.0,
    "acquisition_cost": 200.0,
    "channels": ["YouTube", "Instagram"]
  }'
```

**Expected Response (`HTTP 200`):**
```json
{
  "forecasted_revenue": 195142.05,
  "profitable": true,
  "status": "PROFITABLE"
}
```

### 3.3 Vectorized Batch Revenue Forecasting (`POST /forecast_revenue/batch`)

Demonstrate evaluating multi-brand candidate campaigns in a single vectorized matrix inference call ([ADR-0004](DECISIONS.md#adr-0004-vectorized-batch-inference-architecture)):

```bash
curl -X POST http://127.0.0.1:8000/forecast_revenue/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"brand": "nykaa", "impressions": 50000, "clicks": 4000, "leads": 1500, "conversions": 500},
      {"brand": "purplle", "impressions": 40000, "clicks": 3000, "leads": 1000, "conversions": 300},
      {"brand": "tira", "impressions": 60000, "clicks": 5000, "leads": 2000, "conversions": 800}
    ]
  }'
```

**Expected Response (`HTTP 200`):**
```json
{
  "predictions": [275122.22, 195142.05, 415705.16],
  "total_items": 3,
  "latency_ms": 28.58
}
```

### 3.4 Vectorized Batch Profitability Classification (`POST /predict_profitability/batch`)

```bash
curl -X POST http://127.0.0.1:8000/predict_profitability/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"brand": "nykaa", "impressions": 50000, "clicks": 4000, "leads": 1500, "conversions": 500},
      {"brand": "purplle", "impressions": 40000, "clicks": 3000, "leads": 1000, "conversions": 300},
      {"brand": "tira", "impressions": 60000, "clicks": 5000, "leads": 2000, "conversions": 800}
    ]
  }'
```

**Expected Response (`HTTP 200`):**
```json
{
  "predictions": [
    {"forecasted_revenue": 275122.22, "profitable": true, "status": "PROFITABLE"},
    {"forecasted_revenue": 195142.05, "profitable": true, "status": "PROFITABLE"},
    {"forecasted_revenue": 415705.16, "profitable": true, "status": "PROFITABLE"}
  ],
  "total_items": 3,
  "latency_ms": 38.60
}
```

---

## 4. Strict Invariant Boundary Enforcement (30 sec)

Demonstrate perimeter defense against logically impossible inputs using Pydantic v2 cross-field validation ([ADR-0003](DECISIONS.md#adr-0003-strict-pydantic-v2-invariant-boundary-validation)).

Submit a request where **clicks exceed impressions** (`clicks: 60000`, `impressions: 50000`):

```bash
curl -X POST http://127.0.0.1:8000/forecast_revenue \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "nykaa",
    "impressions": 50000,
    "clicks": 60000
  }'
```

**Expected Response (`HTTP 422 Unprocessable Entity`):**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, Logical invariant violated: clicks (60000.0) cannot exceed impressions (50000.0)"
    }
  ]
}
```

> **Why this matters:** The ML inference pipeline is protected from corrupt out-of-distribution data, and platform engineers receive immediate, actionable diagnostics instead of silent errors or misleading predictions.

---

## 5. Automated End-to-End Smoke Verification (30 sec)

Execute the complete end-to-end smoke verification test suite in offline mock mode or against the live server:

### In-Process Offline Mode (FastAPI TestClient in Lifespan Context)

```bash
python scripts/smoke_test.py --mock
```

### Live Running Server Mode

```bash
python scripts/smoke_test.py --url http://127.0.0.1:8000
```

**Expected Terminal Output:**
```text
==============================================================================
  MARKETING CAMPAIGN INTELLIGENCE - END-TO-END SMOKE VERIFICATION
==============================================================================
 Execution Mode: In-Process FastAPI TestClient (Lifespan Context)
------------------------------------------------------------------------------
 [PASS] 1. Operational Health Probe (GET /health) (6.3ms)
        Detail: status='ok', version='1.0.0', uptime=0.29s, models_loaded=(revenue=True, profit=True)
 [PASS] 2. Readiness Probe (GET /ready) (3.2ms)
        Detail: status='ready', models_loaded=True
 [PASS] 3. Dark-Mode Glassmorphic Web App (GET /app) (4.8ms)
        Detail: status=200, content-type='text/html; charset=utf-8', verified markers: Campaign Intelligence, Nykaa, Purplle, Tira
 [PASS] 4. Single Revenue Forecasting (POST /forecast_revenue) (29.2ms)
        Detail: forecasted_revenue=INR 275,122.22
 [PASS] 5. Single Profitability Prediction (POST /predict_profitability) (21.4ms)
        Detail: forecasted_revenue=INR 275,122.22, profitable=True, status='PROFITABLE'
 [PASS] 6. Batch Revenue Forecasting (POST /forecast_revenue/batch) (33.5ms)
        Detail: total_items=3, predictions=[INR 275,122, INR 195,142, INR 415,705], latency=28.58ms
 [PASS] 7. Batch Profitability Prediction (POST /predict_profitability/batch) (42.0ms)
        Detail: total_items=3, classifications=['PROFITABLE', 'PROFITABLE', 'PROFITABLE'], latency=38.6ms
 [PASS] 8. Boundary Invariant Rejection (clicks > impressions -> HTTP 422) (2.3ms)
        Detail: Correctly rejected with HTTP 422 and logical invariant violation error
------------------------------------------------------------------------------
 SUMMARY: 8/8 passed | 0 failed | Total latency: 142.8ms
 [SUCCESS] All microservice operational & ML inference contracts verified!
==============================================================================
```

---

## 6. Evaluation Reports & EDA Visualizations

Inspect the offline model validation metrics and visual distribution plots:

```bash
# View holdout split evaluation metrics (R², RMSE, weighted F1)
type reports\evaluation.md
```

Explore exploratory data analysis visual artifacts located in `docs/eda/`:
- `roi_distribution.png`: ROI spread across multi-brand campaigns.
- `correlation_heatmap.png`: Feature relationships and multicollinearity assessment.
- `roi_by_campaign_type.png`: Comparative performance by marketing strategy.

---

## Showcase Checklist

- [ ] Single Uvicorn microservice starts cleanly on port 8000 (`uvicorn api.main:app`).
- [ ] Liveness probe (`GET /health`) returns HTTP 200 with uptime and model booleans.
- [ ] Readiness probe (`GET /ready`) returns HTTP 200 with `models_loaded: true`.
- [ ] Interactive glassmorphic SPA at `http://127.0.0.1:8000/app` runs revenue and profitability predictions dynamically.
- [ ] Root redirect `GET /` redirects to `/app` with HTTP 307.
- [ ] Vectorized batch endpoints (`/batch`) process multi-record payloads in <50ms.
- [ ] Boundary invariants reject invalid inputs (`clicks > impressions`) with HTTP 422.
- [ ] Automated smoke verification script (`python scripts/smoke_test.py --mock`) passes all 8 checks with exit code 0.
- [ ] Zero Streamlit runtime dependencies or port 8501 references remain.
