# Architecture Decision Records
Permanent record of foundational architectural decisions, trade-offs, and governance policies for Marketing Campaign Performance Analysis.

*Related Documents: [Architecture](ARCHITECTURE.md), [Specification](SPEC.md), [Mental Model](PROJECT_MENTAL_MODEL.md)*

---

## Index

| ID | Title | Status | Date |
|:---|:------|:-------|:-----|
| [ADR-0001](#adr-0001-technology-stack-and-container-topology) | Technology Stack and Container Topology | Superseded by ADR-0007 | 2026-10-05 |
| [ADR-0002](#adr-0002-in-memory-model-lifecycle-management-via-fastapi-lifespan) | In-Memory Model Lifecycle Management via FastAPI Lifespan | Accepted | 2026-10-05 |
| [ADR-0003](#adr-0003-strict-pydantic-v2-invariant-boundary-validation) | Strict Pydantic v2 Invariant Boundary Validation | Accepted | 2026-10-05 |
| [ADR-0004](#adr-0004-vectorized-batch-inference-architecture) | Vectorized Batch Inference Architecture | Accepted | 2026-10-05 |
| [ADR-0005](#adr-0005-operational-liveness-and-readiness-probes-separation) | Operational Liveness and Readiness Probes Separation | Accepted | 2026-10-05 |
| [ADR-0006](#adr-0006-backward-compatibility-for-scikit-learn-model-unpickling) | Backward Compatibility for Scikit-Learn Model Unpickling | Accepted | 2026-10-05 |
| [ADR-0007](#adr-0007-elimination-of-streamlit-in-favor-of-unified-fastapi-glassmorphic-web-application) | Elimination of Streamlit in Favor of Unified FastAPI Glassmorphic Web Application | Accepted | 2026-10-05 |

---

## ADR-0001: Technology Stack and Container Topology
Date: 2026-10-05  
Status: Superseded by ADR-0007

### Context
The marketing campaign performance platform started as a local data-science exploration project. To support enterprise growth teams evaluating multi-brand ad campaigns across Nykaa, Purplle, and Tira, the platform needs to operate as a high-availability, low-latency microservice while continuing to offer an interactive simulator dashboard for business stakeholders.

### Decision
We will maintain the core Python stack (FastAPI for the REST API microservice and Streamlit for the visual executive dashboard) and package the application into a standardized multi-stage Docker container. The container will be deployable locally via Docker Compose and into cloud environments using Google Cloud Run (`asia-south1`). The API and the dashboard remain cleanly separated so that high-volume API inference does not compete for resources with human dashboard browsing.

### Alternatives considered
- *Monolithic Streamlit deployment:* Running everything inside a single Streamlit process. This was rejected because Streamlit is not designed to serve high-concurrency automated machine learning requests or batch bidding traffic.
- *Flask or Django REST Framework:* Replaced by FastAPI because FastAPI provides native asynchronous concurrency, automated OpenAPI documentation, and high-performance Pydantic data validation out of the box.
- *Heavyweight ML serving engines (e.g., Triton or TorchServe):* Overkill for tabular Scikit-learn and XGBoost pipelines, which run efficiently directly in Python without specialized C++ serving infrastructure.

### Consequences
- Clean separation of concerns between visual scenario planning and machine-to-machine API traffic.
- Standardized container images that deploy identically on developer workstations, CI pipelines, and cloud runners.
- Cost efficiency: The serverless Cloud Run container can scale down to zero when inactive, avoiding persistent cloud infrastructure expenses.

---

## ADR-0002: In-Memory Model Lifecycle Management via FastAPI Lifespan
Date: 2026-10-05  
Status: Accepted

### Context
In the prototype API, machine learning models (`revenue_regressor.joblib` and `profit_classifier.joblib`) were read from disk on every incoming HTTP request. Reading binary model files from disk repeatedly causes high latency (50ms to 150ms per request), exhausts disk I/O, and causes memory instability under concurrent requests.

### Decision
We will use FastAPI's asynchronous `lifespan` context manager to load both the XGBoost regression pipeline and classification pipeline into memory (`app.state.models`) once during application startup. All subsequent requests read directly from the in-memory singletons, with zero disk reads during request processing.

### Alternatives considered
- *Loading models globally on module import:* This causes unit tests to fail or hang when testing individual components, slows down CLI startup, and complicates mocking.
- *Lazy loading on first request:* Causes an unpredictable latency spike ("cold start") for the first unfortunate user or bidding engine that hits the service.
- *Maintaining existing per-request loading:* Unacceptable for production due to excessive disk I/O and poor response times.

### Consequences
- Request processing latency drops from ~100ms to under 5ms.
- Predictable container startup: the service only accepts traffic after models are confirmed to be loaded and healthy in memory.
- In test environments, mock models can be cleanly injected into `app.state` without touching the filesystem.

---

## ADR-0003: Strict Pydantic v2 Invariant Boundary Validation
Date: 2026-10-05  
Status: Accepted

### Context
Ad campaign parameters submitted by third-party systems or users can contain logical contradictions or invalid numbers (for example, reporting more clicks than impressions, more conversions than clicks, negative spending, or unrecognized brands). In the prototype, these inputs could cause unrealistic revenue forecasts or runtime zero-division crashes.

### Decision
We will enforce strict boundary and invariant validation at the API edge using Pydantic v2. Field constraints ensure metrics are non-negative, execution months fall between 1 and 12, and brands are case-insensitively matched to supported brands (`nykaa`, `purplle`, `tira`). Cross-field model validators enforce domain logic: clicks cannot exceed impressions, conversions cannot exceed clicks, and leads cannot exceed clicks. Any violation immediately returns an explicit HTTP 422 Unprocessable Entity error with clear diagnostic messages before ML pipelines are executed.

### Alternatives considered
- *Silent correction (clamping values):* Automatically capping clicks to impressions. This was rejected because it hides upstream data bugs from advertisers and platform engineers.
- *In-pipeline assertions:* Allowing invalid requests to reach model code before raising exceptions. This leads to unhelpful HTTP 500 internal server errors.
- *Permissive schemas:* Accepting raw data and letting ML models predict on invalid inputs, producing misleading business forecasts.

### Consequences
- Bad data is rejected immediately at the perimeter with actionable error messages.
- The machine learning pipelines are protected from out-of-distribution invalid inputs.
- Zero-safe calculations in feature engineering guarantee that edge cases like zero impressions or zero clicks evaluate safely without runtime crashes.

---

## ADR-0004: Vectorized Batch Inference Architecture
Date: 2026-10-05  
Status: Accepted

### Context
Growth marketers and programmatic bidding pipelines regularly evaluate large batches of candidate campaign configurations (e.g., comparing 100 variations of audience, channel mix, and cost). Sending individual HTTP requests for each campaign introduces excessive network latency and connection overhead.

### Decision
We will introduce `/forecast_revenue/batch` and `/predict_profitability/batch` endpoints accepting arrays of 1 to 500 campaign records in a single HTTP POST request. Feature engineering in `src/inference.py` will vectorize the conversion of multiple campaign objects directly into single batch DataFrames, allowing Scikit-learn and XGBoost to perform vectorized matrix inference in a single call.

### Alternatives considered
- *Sequential loops over single-item inference:* Iterating through records one by one in Python. This is significantly slower because it incurs per-row Pandas overhead.
- *Asynchronous background job queue (Celery + Redis):* Overcomplicated for synchronous batches under 500 items, adding operational complexity and database dependencies.
- *Unbounded batch sizes:* Permitting unlimited items. This was rejected to prevent memory exhaustion and denial-of-service vulnerabilities; batches are capped at 500 items.

### Consequences
- Batches of 500 campaigns evaluate in under 50 milliseconds.
- Reduced network round-trips for automated client pipelines.
- Bounded memory usage ensures predictable resource consumption on container instances.

---

## ADR-0005: Operational Liveness and Readiness Probes Separation
Date: 2026-10-05  
Status: Accepted

### Context
Cloud orchestration platforms (like Kubernetes and Google Cloud Run) distinguish between whether a process is running (liveness) and whether it is ready to receive live user traffic (readiness). A single `/health` endpoint that checks filesystem paths cannot confirm whether machine learning models are actively loaded into memory.

### Decision
We will split system observability into two distinct probes:
1. `GET /health` (Liveness): Returns process uptime, version, and basic operational health.
2. `GET /ready` (Readiness): Verifies that both the revenue regressor and profitability classifier are successfully loaded and warmed in `app.state.models`. Returns HTTP 200 when ready, or HTTP 503 Service Unavailable if models are missing or uninitialized.
Additionally, a lightweight middleware will emit structured JSON logs with HTTP status and response latency for every request.

### Alternatives considered
- *A single combined `/health` endpoint:* Can cause race conditions where a newly started container receives live customer traffic before model weights have finished loading.
- *No readiness probe:* Relies solely on TCP port checks, routing traffic to unready containers and returning 500 errors to users.

### Consequences
- Cloud load balancers only route traffic to fully initialized containers, eliminating startup error spikes.
- Full backwards compatibility is preserved for existing monitoring tools calling `GET /health`.
- Structured JSON logs enable seamless aggregation in Cloud Logging or Datadog.

---

## ADR-0006: Backward Compatibility for Scikit-Learn Model Unpickling
Date: 2026-10-05  
Status: Accepted

### Context
The trained model binaries (`revenue_regressor.joblib` and `profit_classifier.joblib`) were serialized using Scikit-learn with internal column transformer classes (specifically `_RemainderColsList`). In certain newer or older Scikit-learn minor releases, unpickling this object directly can fail with an `AttributeError` if the internal class is not defined.

### Decision
We will maintain a lightweight compatibility class shim for `sklearn.compose._column_transformer._RemainderColsList` in shared inference loading points across the API and dashboard. If the class is missing from the active Scikit-learn installation, the shim defines a subclass of `list` to allow safe deserialization without retraining the models.

### Alternatives considered
- *Retraining all models from scratch:* Discarding existing trained weights. Rejected because existing models are frozen, validated, and meet accuracy criteria.
- *Pinning an exact single build of Scikit-learn without shims:* Fragile across different base operating system images and Python minor versions.

### Consequences
- Serialized model pipelines load reliably across environments without runtime deserialization errors.
- Existing pre-trained models remain fully functional without expensive retraining or pipeline re-export.

---

## ADR-0007: Elimination of Streamlit in Favor of Unified FastAPI Glassmorphic Web Application
Date: 2026-10-05  
Status: Accepted

### Context
The initial architecture plan (ADR-0001) paired a FastAPI REST microservice with an independent Streamlit server for the executive dashboard. In practice, running two separate Python runtimes (FastAPI on port 8000 and Streamlit on port 8501) introduces significant operational complexity: it requires dual container orchestration or multi-process management, consumes over 250 MB of unnecessary memory overhead, complicates reverse proxies, and is incompatible with single-port serverless containers on Google Cloud Run. In contrast, the user's top production systems (CustomerSupportAnalytics and FinancialIntelligenceCopilot) utilize a unified single-process architecture where FastAPI serves a zero-dependency, dark-mode glassmorphic single-page web application directly at `/app` via `api/index.html`.

### Decision
We will eliminate Streamlit from the project dependencies and runtime architecture entirely. FastAPI will serve a self-contained, dark-mode glassmorphic single-page web application directly at `GET /app` (with root redirect `GET /` -> `/app`) using `fastapi.responses.FileResponse` pointing to `api/index.html`. The frontend will operate entirely client-side, interacting with the backend via native asynchronous `fetch()` calls to `/forecast_revenue`, `/predict_profitability`, and `/health`. The entire service collapses into a single unified Uvicorn container process listening on a single port (`${PORT:-8000}`).

### Alternatives considered
- *Dual-container Docker Compose / multi-service Cloud Run:* Running FastAPI and Streamlit as separate cloud services. Rejected due to doubled infrastructure cost, cold-start latencies, and cross-origin resource sharing (CORS) complexity.
- *Multi-process container with Supervisord:* Running both FastAPI and Streamlit inside a single container image managed by Supervisord. Rejected because it violates the single-concern container principle, complicates signal handling (SIGTERM/SIGINT), and makes health checking fragile.
- *Full React / Next.js / Vue framework SPA:* Introduces Node.js, npm build pipelines, bundle compilation, and deployment complexity. Rejected because a static vanilla HTML/CSS/JS glassmorphic SPA delivers an identical high-fidelity aesthetic without any build-step dependencies.

### Consequences
- **Unified Container Topology:** The entire application runs as a single Uvicorn ASGI process within one lightweight Docker container, natively compliant with Google Cloud Run and local execution.
- **Resource Efficiency:** Eliminating Streamlit removes heavy dependencies (such as Tornado, Altair, pyarrow, and protobuf) from the production image, cutting container memory usage by >50% and shaving startup latency.
- **Zero-Build Glassmorphic UI:** The dark-mode glassmorphic interface (`backdrop-filter: blur(16px)`, radial glow effects, responsive parameter sliders, and real-time INR currency formatting) lives in `api/index.html`, editable and inspectable without Node.js or Webpack.
- **Single Port Simplicity:** Eliminates dual-port configuration; all REST API endpoints, health probes, and the interactive UI are accessed through the primary HTTP port (e.g., 8000).
