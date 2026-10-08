# Marketing Campaign Intelligence Production Hardening Specification
A comprehensive specification for production hardening, lifespan model serving, batch inference, glassmorphic UI enhancements, and operational readiness for the Marketing Campaign Performance Analysis platform.

*Date: 2026-10-05*  
*Related Documents: [Mental Model](PROJECT_MENTAL_MODEL.md), [Codebase Map](CODEBASE_MAP.md)*

---

## 1. Summary

- **Vision:** Transform the existing Marketing Campaign Performance Analysis system from a local prototype into an enterprise-ready, high-throughput machine learning microservice and executive presentation dashboard for multi-brand e-commerce marketing portfolios (Nykaa, Purplle, Tira).
- **Posture:** **Production**. Zero unhandled exceptions, zero silent data corruption, strict Pydantic v2 boundary schema validation, in-memory model lifecycle management (eliminating per-request disk reads), production containerization, automated CI testing, and observability.
- **Target Persona:**
  - **Primary:** Growth Marketing Leads, Performance Marketers, and FinOps Analysts evaluating multi-brand ad spend allocations, forecasting campaign revenue, and screening for campaign profitability before committing media budgets.
  - **Secondary:** ML / Platform Engineers integrating automated campaign revenue forecasting and profitability classification into ad bidding pipelines or enterprise BI dashboards.

---

## 2. User Journeys

The platform already provides single-record REST endpoints (`/forecast_revenue`, `/predict_profitability`) and a standard 3-column Streamlit dashboard. The following journeys describe new and enhanced capabilities starting from that existing baseline.

### Journey 1: High-Throughput Batch Campaign Revenue Forecasting & Profitability Evaluation
*Starting from:* Existing single-record REST endpoints (`POST /forecast_revenue` and `POST /predict_profitability` in [`api/main.py`](../api/main.py)).
1. **API Client Submits Batch Request:** A platform engineer or automated bidding pipeline dispatches an HTTP `POST` request to `/forecast_revenue/batch` or `/predict_profitability/batch` with a JSON payload containing an array of up to 500 campaign parameter records.
2. **Lifespan In-Memory Model Execution:** The FastAPI service delegates inference directly to pre-loaded in-memory XGBoost model singletons stored on `app.state`, completely bypassing disk I/O.
3. **Batch Response Delivery:** The API client receives an HTTP 200 response containing an ordered list of campaign prediction objects, total item count, and inference latency metadata.
4. **Predictive Analytics Consumption:** Downstream systems ingest the structured predictions to automate media budget allocation across candidate campaigns.

### Journey 2: Executive Glassmorphic Web Experience & Interactive Campaign Simulation
*Starting from:* Replacing prototype UI with a modern glassmorphic web application served directly by FastAPI at `/app` ([`api/index.html`](../api/index.html)).
1. **Stakeholder Navigates to Dashboard:** A growth marketing executive accesses `GET /app` (or root `GET /` which redirects to `/app`) and observes a modern dark-mode glassmorphic interface featuring frosted-glass panels (`backdrop-filter: blur(16px)`), radial glowing gradients, branded badge indicators for Nykaa, Purplle, and Tira, and live microservice health pill.
2. **Interactive Parameter Configuration:** The marketer configures ad parameters across a 3-column layout: Brand selector, Campaign Strategy, Target Audience, Customer Segment, Execution Month slider (1–12), Expected Impressions, Click volume, Engagement score, Projected Leads, Conversions, Cost Per Acquisition (CPA), and multi-select Delivery Channels.
3. **Execution & Dynamic Projection:** The marketer clicks "🚀 Calculate Campaign Forecasts". The web app issues asynchronous client-side `fetch()` requests to `POST /forecast_revenue` and `POST /predict_profitability` on the same host, displaying animated KPI scorecards.
4. **Insight Inspection:** The marketer inspects the primary KPI cards displaying Forecasted Revenue (formatted in INR currency `₹xx,xxx.xx`), Profitability Status (vibrant emerald badge for `PROFITABLE CAMPAIGN` or ruby badge for `NET OPERATIONAL LOSS`), alongside derived efficiency metrics (CTR, Conversion Rate, Cost Per Lead).

### Journey 3: Microservice Lifecycle, Probes & Operational Observability
*Starting from:* Existing simple `/health` endpoint performing synchronous `os.path.exists()` checks against local model files.
1. **Container Startup & Lifespan Initialization:** The microservice starts via Uvicorn within Docker. The lifespan handler loads `models/revenue_regressor.joblib` and `models/profit_classifier.joblib` into memory once, warming the inference pipelines.
2. **Kubernetes / Cloud Run Readiness Check:** The orchestrator invokes `GET /ready`. The API inspects `app.state.models`, confirms pipeline readiness, and responds with HTTP 200 and readiness status.
3. **Continuous Liveness Monitoring:** An uptime monitor or load balancer polls `GET /health` periodically, receiving HTTP 200 with service version, uptime seconds, model availability flags, and operational status.
4. **Structured Request Logging:** During ongoing inference, every request emits structured JSON logs containing timestamp, endpoint, HTTP status, and request duration without leaking sensitive parameters.

### Journey 4: Resilient Edge-Case & Invariant Boundary Validation
*Starting from:* Existing permissive Pydantic schema allowing logically inconsistent inputs (e.g., clicks exceeding impressions, negative spend, unknown brands).
1. **Client Sends Boundary Violations:** A client or integration pipeline submits a campaign request containing edge-case values (e.g., zero impressions, clicks > impressions, negative CPA, or unlisted brand name).
2. **Strict Schema Interception:** Pydantic v2 model validators immediately intercept logical invariant violations (e.g., `clicks > impressions` or `conversions > clicks`) and invalid brands before any feature transformation or model inference occurs.
3. **Clear Diagnostic Feedback:** The client receives an HTTP 422 Unprocessable Entity response containing explicit field-level error messages describing the invariant violated and listing supported options.
4. **Safe Handling of Zero Divisions:** When valid zero-boundary values are submitted (e.g., `impressions = 0`, `clicks = 0`, `leads = 0`), the feature builder safely sets `ctr = 0.0`, `conversion_rate = 0.0`, and `cpl = 0.0`, returning valid model forecasts without runtime errors or `NaN` values.

---

## 3. Acceptance Criteria

### High-Throughput Batch Inference (Journey 1)
- **AC-BATCH-01:** Given a running FastAPI service with pre-loaded models, when an HTTP `POST` request is sent to `/forecast_revenue/batch` with a JSON payload containing 5 valid campaign items, then the service returns HTTP 200 with a response containing `predictions` (a list of 5 floats), `total_items: 5`, and each predicted value matching the individual `/forecast_revenue` output for the corresponding item.
- **AC-BATCH-02:** Given a running FastAPI service with pre-loaded models, when an HTTP `POST` request is sent to `/predict_profitability/batch` with a JSON payload containing 3 valid campaign items, then the service returns HTTP 200 with `predictions` (a list of 3 items containing `forecasted_revenue`, `profitable`, and `status`), and `total_items: 3`.
- **AC-BATCH-03:** Given a batch request to `/forecast_revenue/batch` or `/predict_profitability/batch`, when the payload contains an empty array `[]`, then the service returns HTTP 422 Unprocessable Entity with a validation message indicating that the batch list must contain at least 1 item.
- **AC-BATCH-04:** Given a batch request containing 10 items where item index 3 has `clicks > impressions`, then the service returns HTTP 422 Unprocessable Entity specifying index 3 and the invariant failure details, and no predictions are computed.
- **AC-BATCH-05:** Given a batch request exceeding the maximum limit of 500 items, when submitted to any batch endpoint, then the service returns HTTP 422 Unprocessable Entity rejecting the oversized payload.

### Glassmorphic Web Experience & Interactive Simulation (Journey 2)
- **AC-UI-01:** Given the web application is served at `GET /app` from `api/index.html`, when inspecting the styles, then container styling includes `backdrop-filter: blur(16px)`, semi-transparent background (`rgba(...)`), radial glowing gradients, and rounded corner borders (`border-radius >= 12px`).
- **AC-UI-02:** Given valid campaign inputs on the glassmorphic dashboard, when the user clicks "🚀 Calculate Campaign Forecasts", then the app dispatches client-side `fetch()` requests and displays the Forecasted Revenue formatted as Indian Rupee currency (e.g., `₹125,000.00`) and the Profitability Status within 1.0 second.
- **AC-UI-03:** Given campaign inputs resulting in a profitable prediction (`profitable == true`), when forecasts are calculated, then the status card displays `📈 Forecasted Status: PROFITABLE CAMPAIGN` with emerald glass styling.
- **AC-UI-04:** Given campaign inputs resulting in an unprofitable prediction (`profitable == false`), when forecasts are calculated, then the status card displays `📉 Forecasted Status: NET OPERATIONAL LOSS` with ruby glass styling.
- **AC-UI-05:** Given the user accesses root path `GET /`, then the service returns an HTTP 307 temporary redirect to `/app`.

### Microservice Lifecycle, Probes & Observability (Journey 3)
- **AC-LIFE-01:** Given the FastAPI microservice initializes during startup lifespan, when both `models/revenue_regressor.joblib` and `models/profit_classifier.joblib` exist, then both pipelines are loaded into `app.state` as in-memory singletons and no disk I/O is performed during subsequent `/forecast_revenue` or `/predict_profitability` requests.
- **AC-PROBE-01:** Given the microservice has finished startup initialization, when `GET /ready` is called, then the service returns HTTP 200 with `{"status": "ready", "models_loaded": true}`.
- **AC-PROBE-02:** Given one or both model artifacts are missing or unreadable at startup, when `GET /ready` is called, then the service returns HTTP 503 Service Unavailable with `{"status": "unready", "models_loaded": false}` and descriptive error details.
- **AC-PROBE-03:** Given a running microservice, when `GET /health` is called, then the service returns HTTP 200 with `{"status": "ok", "version": "1.0.0", "revenue_model": true, "profit_model": true, "uptime_seconds": <float>}`.
- **AC-OBS-01:** Given an HTTP request to any API endpoint, when the request completes, then a structured log entry is generated recording the method, path, status code, and latency in milliseconds.

### Validation Invariants & Error Edge Cases (Journey 4)
- **AC-ERR-01 (Negative Metric Invariants):** Given a request to `/forecast_revenue` or `/predict_profitability` with any of `impressions < 0`, `clicks < 0`, `leads < 0`, `conversions < 0`, `acquisition_cost < 0`, or `engagement_score < 0`, then the service returns HTTP 422 Unprocessable Entity specifying the negative field.
- **AC-ERR-02 (Logical Invariant Clicks <= Impressions):** Given a request where `clicks > impressions` (e.g., `impressions: 100`, `clicks: 150`), then the service returns HTTP 422 Unprocessable Entity with a validation error indicating that clicks cannot exceed impressions.
- **AC-ERR-03 (Logical Invariant Conversions <= Clicks):** Given a request where `conversions > clicks` (e.g., `clicks: 50`, `conversions: 80`), then the service returns HTTP 422 Unprocessable Entity with a validation error indicating that conversions cannot exceed clicks.
- **AC-ERR-04 (Logical Invariant Leads <= Clicks):** Given a request where `leads > clicks` (e.g., `clicks: 100`, `leads: 120`), then the service returns HTTP 422 Unprocessable Entity with a validation error indicating that leads cannot exceed clicks.
- **AC-ERR-05 (Month Range):** Given a request where `month < 1` or `month > 12`, then the service returns HTTP 422 Unprocessable Entity with a validation error indicating month must be between 1 and 12 inclusive.
- **AC-ERR-06 (Brand Categorical Constraint):** Given a request where `brand` is not one of `['nykaa', 'purplle', 'tira']` (e.g., `"brand": "sephora"`), then the service returns HTTP 422 Unprocessable Entity with detail specifying valid brands `['nykaa', 'purplle', 'tira']`.
- **AC-ERR-07 (Brand Case Normalization):** Given a request where `brand` is submitted with mixed or uppercase characters (e.g., `"brand": "Nykaa"` or `"brand": "PURPLLE"`), then the service normalizes the value to lowercase and processes the request with HTTP 200 without error.
- **AC-ERR-08 (Zero-Safe CTR):** Given a campaign input where `impressions == 0` and `clicks == 0`, when processed by `build_campaign_row`, then `ctr` evaluates to `0.0` without raising `ZeroDivisionError` or producing `NaN`.
- **AC-ERR-09 (Zero-Safe Conversion Rate):** Given a campaign input where `clicks == 0` and `conversions == 0`, when processed by `build_campaign_row`, then `conversion_rate` evaluates to `0.0` without raising `ZeroDivisionError` or producing `NaN`.
- **AC-ERR-10 (Zero-Safe CPL):** Given a campaign input where `leads == 0`, when processed by `build_campaign_row`, then `cpl` evaluates to `0.0` without raising `ZeroDivisionError` or producing `NaN`.

---

## 4. Regression Criteria

The following behaviors and contracts currently supported by the platform must remain fully functional after all changes:

- **RC-01 (Single Forecast Revenue Endpoint):** Given the existing contract for `POST /forecast_revenue`, when called with a valid `CampaignInput` body (e.g., `{"brand": "nykaa"}`), then the endpoint returns HTTP 200 and a JSON object containing `"forecasted_revenue": <float>`.
- **RC-02 (Single Profitability Endpoint):** Given the existing contract for `POST /predict_profitability`, when called with a valid `CampaignInput` body (e.g., `{"brand": "nykaa"}`), then the endpoint returns HTTP 200 with `"forecasted_revenue": <float>`, `"profitable": <bool>`, and `"status": "PROFITABLE" | "LOSS"`.
- **RC-03 (Health Check Compatibility):** Given existing clients calling `GET /health`, when invoked, then the response contains keys `"status"`, `"revenue_model"`, and `"profit_model"` with expected boolean values.
- **RC-04 (Campaign Feature Engineering Contract):** Given a dictionary payload passed to `src.inference.build_campaign_row(payload)`, when executed, then it returns a 2-tuple `(df_reg, df_cls)` where both DataFrames contain all 6 one-hot channel columns (`channel_youtube`, `channel_instagram`, `channel_google`, `channel_whatsapp`, `channel_email`, `channel_facebook`), cyclical `month_sin`/`month_cos`, and calculated derived metrics (`ctr`, `conversion_rate`, `cpl`).
- **RC-05 (Default Inputs Stability):** Given a `CampaignInput` initialized with default parameters, when dumped to dictionary and evaluated, then all fields (`brand="nykaa"`, `campaign_type="Paid Ads"`, `month=5`, `impressions=50000.0`, etc.) produce a valid prediction without schema validation failure.
- **RC-06 (Baseline Pytest Suite):** Given the 10 existing unit and API tests in `tests/test_api.py` and `tests/test_inference.py`, when `pytest` is executed, then all 10 tests continue to pass without regression.

---

## 5. Non-Goals

1. **No Model Retraining:** The existing trained XGBoost pipelines (`models/revenue_regressor.joblib` and `models/profit_classifier.joblib`) are frozen and reused as-is. No hyperparameter tuning or retraining will take place.
2. **No External Live Ad Connectors:** No live API sync or webhooks connecting to Google Ads API, Meta Marketing API, or TikTok Ads API.
3. **No Authentication or Multi-Tenancy:** The service operates as a single-tenant showcase microservice without OAuth2, JWT tokens, or API key rate-limiting.
4. **No Real-Time Streaming or WebSockets:** Inference is strictly request-response over HTTP REST; no WebSockets or streaming inference channels will be built.
5. **No Database Write Operations on Inference:** Model serving does not write prediction logs or transaction records back to PostgreSQL during request processing.

---

## 6. Open Questions & Assumptions

1. **Batch Endpoint Payload Size Limit:**
   - *Ambiguity:* The mental model specifies bulk batch prediction endpoints for high-throughput campaign evaluation, but does not define an upper bound for a single batch request.
   - *Assumption:* A single batch request is bounded to a maximum of 500 items (`concurrency / memory safeguard`). Payloads exceeding 500 records will be rejected with HTTP 422.
2. **Model Lifespan Test Fixture Strategy:**
   - *Ambiguity:* In `tests/test_api.py`, `_load_model` was patched per-test. When lifespan is implemented, loading occurs on application startup before requests arrive.
   - *Assumption:* The test client in `tests/test_api.py` will use `TestClient(app)` with FastAPI's lifespan context or inject mock models into `app.state.models` during fixture setup, ensuring isolation and preventing real model disk loads during unit testing if desired.
3. **Pydantic V2 Model Validator Compatibility with Older Callers:**
   - *Ambiguity:* The existing `CampaignInput` had default values for all fields. Will partial payloads still be allowed?
   - *Assumption:* All fields retain sensible default values matching the current baseline, allowing partial payloads (e.g., `{"brand": "tira"}`) to remain valid while enforcing strict logical invariant rules whenever metrics are explicitly provided.
