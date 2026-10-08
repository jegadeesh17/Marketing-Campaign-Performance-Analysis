"""End-to-End Smoke Verification Script for Marketing Campaign Intelligence.

Verifies operational health, readiness, web UI serving, single & batch model
predictions, and boundary invariant validation against a live server or via
in-process FastAPI TestClient lifespan context.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, NamedTuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import httpx

try:
    from fastapi.testclient import TestClient
    from api.main import app
except Exception as e:
    TestClient = None  # type: ignore[assignment, misc]
    app = None  # type: ignore[assignment]


class StepResult(NamedTuple):
    name: str
    passed: bool
    detail: str
    duration_ms: float


def run_smoke_tests(client: Any) -> list[StepResult]:
    """Execute all 8 smoke verification checks against an HTTP client or TestClient."""
    results: list[StepResult] = []

    # -------------------------------------------------------------------------
    # 1. GET /health
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        res = client.get("/health")
        duration = (time.perf_counter() - t0) * 1000.0
        if res.status_code != 200:
            results.append(
                StepResult(
                    name="1. Operational Health Probe (GET /health)",
                    passed=False,
                    detail=f"Expected HTTP 200, got {res.status_code}: {res.text}",
                    duration_ms=duration,
                )
            )
        else:
            data = res.json()
            status_ok = data.get("status") == "ok"
            version_ok = data.get("version") == "1.0.0"
            uptime = data.get("uptime_seconds")
            uptime_ok = isinstance(uptime, (int, float)) and uptime >= 0.0
            models_ok = data.get("revenue_model") is True and data.get("profit_model") is True

            if status_ok and version_ok and uptime_ok and models_ok:
                results.append(
                    StepResult(
                        name="1. Operational Health Probe (GET /health)",
                        passed=True,
                        detail=(
                            f"status='{data.get('status')}', version='{data.get('version')}', "
                            f"uptime={uptime}s, models_loaded=(revenue={data.get('revenue_model')}, "
                            f"profit={data.get('profit_model')})"
                        ),
                        duration_ms=duration,
                    )
                )
            else:
                results.append(
                    StepResult(
                        name="1. Operational Health Probe (GET /health)",
                        passed=False,
                        detail=f"Health payload check failed: {data}",
                        duration_ms=duration,
                    )
                )
    except Exception as exc:
        results.append(
            StepResult(
                name="1. Operational Health Probe (GET /health)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # -------------------------------------------------------------------------
    # 2. GET /ready
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        res = client.get("/ready")
        duration = (time.perf_counter() - t0) * 1000.0
        if res.status_code != 200:
            results.append(
                StepResult(
                    name="2. Readiness Probe (GET /ready)",
                    passed=False,
                    detail=f"Expected HTTP 200, got {res.status_code}: {res.text}",
                    duration_ms=duration,
                )
            )
        else:
            data = res.json()
            ready_ok = data.get("status") == "ready" and data.get("models_loaded") is True
            if ready_ok:
                results.append(
                    StepResult(
                        name="2. Readiness Probe (GET /ready)",
                        passed=True,
                        detail=f"status='ready', models_loaded={data.get('models_loaded')}",
                        duration_ms=duration,
                    )
                )
            else:
                results.append(
                    StepResult(
                        name="2. Readiness Probe (GET /ready)",
                        passed=False,
                        detail=f"Readiness payload unexpected: {data}",
                        duration_ms=duration,
                    )
                )
    except Exception as exc:
        results.append(
            StepResult(
                name="2. Readiness Probe (GET /ready)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # -------------------------------------------------------------------------
    # 3. GET /app (Glassmorphic SPA)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        res = client.get("/app")
        duration = (time.perf_counter() - t0) * 1000.0
        content_type = res.headers.get("content-type", "")
        body = res.text
        html_ok = "text/html" in content_type
        # Accept either "Marketing Campaign Intelligence" or "Marketing Intelligence"
        has_title = ("marketing campaign intelligence" in body.lower()) or ("marketing intelligence" in body.lower())
        markers = ["Nykaa", "Purplle", "Tira"]
        missing_markers = [m for m in markers if m.lower() not in body.lower()]
        if not has_title:
            missing_markers.append("Marketing [Campaign] Intelligence")

        if res.status_code == 200 and html_ok and not missing_markers:
            results.append(
                StepResult(
                    name="3. Dark-Mode Glassmorphic Web App (GET /app)",
                    passed=True,
                    detail=(
                        f"status=200, content-type='{content_type}', verified markers: "
                        "Campaign Intelligence, Nykaa, Purplle, Tira"
                    ),
                    duration_ms=duration,
                )
            )
        else:
            fail_reasons = []
            if res.status_code != 200:
                fail_reasons.append(f"HTTP status {res.status_code}")
            if not html_ok:
                fail_reasons.append(f"content-type '{content_type}' missing text/html")
            if missing_markers:
                fail_reasons.append(f"missing HTML markers: {missing_markers}")
            results.append(
                StepResult(
                    name="3. Dark-Mode Glassmorphic Web App (GET /app)",
                    passed=False,
                    detail="; ".join(fail_reasons),
                    duration_ms=duration,
                )
            )
    except Exception as exc:
        results.append(
            StepResult(
                name="3. Dark-Mode Glassmorphic Web App (GET /app)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # -------------------------------------------------------------------------
    # 4. POST /forecast_revenue
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    single_payload = {
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
        "channels": ["Instagram", "Google"],
    }
    try:
        res = client.post("/forecast_revenue", json=single_payload)
        duration = (time.perf_counter() - t0) * 1000.0
        if res.status_code != 200:
            results.append(
                StepResult(
                    name="4. Single Revenue Forecasting (POST /forecast_revenue)",
                    passed=False,
                    detail=f"Expected HTTP 200, got {res.status_code}: {res.text}",
                    duration_ms=duration,
                )
            )
        else:
            data = res.json()
            rev = data.get("forecasted_revenue")
            if rev is None:
                rev = data.get("forecasted_revenue_inr")
            if rev is not None and isinstance(rev, (int, float)) and rev > 0:
                results.append(
                    StepResult(
                        name="4. Single Revenue Forecasting (POST /forecast_revenue)",
                        passed=True,
                        detail=f"forecasted_revenue=INR {rev:,.2f}",
                        duration_ms=duration,
                    )
                )
            else:
                results.append(
                    StepResult(
                        name="4. Single Revenue Forecasting (POST /forecast_revenue)",
                        passed=False,
                        detail=f"Revenue prediction non-positive or missing: {data}",
                        duration_ms=duration,
                    )
                )
    except Exception as exc:
        results.append(
            StepResult(
                name="4. Single Revenue Forecasting (POST /forecast_revenue)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # -------------------------------------------------------------------------
    # 5. POST /predict_profitability
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        res = client.post("/predict_profitability", json=single_payload)
        duration = (time.perf_counter() - t0) * 1000.0
        if res.status_code != 200:
            results.append(
                StepResult(
                    name="5. Single Profitability Prediction (POST /predict_profitability)",
                    passed=False,
                    detail=f"Expected HTTP 200, got {res.status_code}: {res.text}",
                    duration_ms=duration,
                )
            )
        else:
            data = res.json()
            rev = data.get("forecasted_revenue")
            if rev is None:
                rev = data.get("forecasted_revenue_inr")
            status = data.get("status") or data.get("predicted_class")
            is_profitable = data.get("profitable")
            prob = data.get("profit_probability")

            valid_rev = rev is not None and isinstance(rev, (int, float)) and rev > 0
            valid_status = status in ["PROFITABLE", "LOSS", "Profitable", "Loss"]
            valid_flag = is_profitable in [True, False]
            valid_prob = True
            if prob is not None:
                valid_prob = isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0

            if valid_rev and valid_status and valid_flag and valid_prob:
                results.append(
                    StepResult(
                        name="5. Single Profitability Prediction (POST /predict_profitability)",
                        passed=True,
                        detail=(
                            f"forecasted_revenue=INR {rev:,.2f}, profitable={is_profitable}, "
                            f"status='{status}'"
                        ),
                        duration_ms=duration,
                    )
                )
            else:
                results.append(
                    StepResult(
                        name="5. Single Profitability Prediction (POST /predict_profitability)",
                        passed=False,
                        detail=f"Profitability prediction fields invalid: {data}",
                        duration_ms=duration,
                    )
                )
    except Exception as exc:
        results.append(
            StepResult(
                name="5. Single Profitability Prediction (POST /predict_profitability)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # -------------------------------------------------------------------------
    # 6. POST /forecast_revenue/batch
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    batch_payload = {
        "items": [
            {
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
                "channels": ["Instagram", "Google"],
            },
            {
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
                "channels": ["YouTube", "Instagram"],
            },
            {
                "brand": "tira",
                "campaign_type": "Influencer",
                "target_audience": "Working Women",
                "language": "English",
                "customer_segment": "Working Women",
                "month": 7,
                "impressions": 60000.0,
                "clicks": 5000.0,
                "leads": 2000.0,
                "conversions": 800.0,
                "engagement_score": 22.0,
                "acquisition_cost": 300.0,
                "channels": ["Instagram", "Facebook"],
            },
        ]
    }
    try:
        res = client.post("/forecast_revenue/batch", json=batch_payload)
        duration = (time.perf_counter() - t0) * 1000.0
        if res.status_code != 200:
            results.append(
                StepResult(
                    name="6. Batch Revenue Forecasting (POST /forecast_revenue/batch)",
                    passed=False,
                    detail=f"Expected HTTP 200, got {res.status_code}: {res.text}",
                    duration_ms=duration,
                )
            )
        else:
            data = res.json()
            total = data.get("total_items")
            if total is None:
                total = data.get("total_processed")
            preds = data.get("predictions", [])
            valid_batch = (
                total == 3
                and len(preds) == 3
                and all(isinstance(p, (int, float)) and p > 0 for p in preds)
            )
            if valid_batch:
                results.append(
                    StepResult(
                        name="6. Batch Revenue Forecasting (POST /forecast_revenue/batch)",
                        passed=True,
                        detail=(
                            f"total_items=3, predictions=[{', '.join(f'INR {p:,.0f}' for p in preds)}], "
                            f"latency={data.get('latency_ms', 0)}ms"
                        ),
                        duration_ms=duration,
                    )
                )
            else:
                results.append(
                    StepResult(
                        name="6. Batch Revenue Forecasting (POST /forecast_revenue/batch)",
                        passed=False,
                        detail=f"Batch revenue response structure invalid: {data}",
                        duration_ms=duration,
                    )
                )
    except Exception as exc:
        results.append(
            StepResult(
                name="6. Batch Revenue Forecasting (POST /forecast_revenue/batch)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # -------------------------------------------------------------------------
    # 7. POST /predict_profitability/batch
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        res = client.post("/predict_profitability/batch", json=batch_payload)
        duration = (time.perf_counter() - t0) * 1000.0
        if res.status_code != 200:
            results.append(
                StepResult(
                    name="7. Batch Profitability Prediction (POST /predict_profitability/batch)",
                    passed=False,
                    detail=f"Expected HTTP 200, got {res.status_code}: {res.text}",
                    duration_ms=duration,
                )
            )
        else:
            data = res.json()
            total = data.get("total_items")
            if total is None:
                total = data.get("total_processed")
            preds = data.get("predictions", [])
            valid_batch = (
                total == 3
                and len(preds) == 3
                and all(
                    isinstance(p, dict)
                    and (p.get("forecasted_revenue") or p.get("forecasted_revenue_inr") or 0) > 0
                    and p.get("profitable") in [True, False]
                    and (p.get("status") in ["PROFITABLE", "LOSS"] or p.get("predicted_class") in ["Profitable", "Loss"])
                    for p in preds
                )
            )
            if valid_batch:
                summary_statuses = [p.get("status") or p.get("predicted_class") for p in preds]
                results.append(
                    StepResult(
                        name="7. Batch Profitability Prediction (POST /predict_profitability/batch)",
                        passed=True,
                        detail=(
                            f"total_items=3, classifications={summary_statuses}, "
                            f"latency={data.get('latency_ms', 0)}ms"
                        ),
                        duration_ms=duration,
                    )
                )
            else:
                results.append(
                    StepResult(
                        name="7. Batch Profitability Prediction (POST /predict_profitability/batch)",
                        passed=False,
                        detail=f"Batch profitability response invalid: {data}",
                        duration_ms=duration,
                    )
                )
    except Exception as exc:
        results.append(
            StepResult(
                name="7. Batch Profitability Prediction (POST /predict_profitability/batch)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # -------------------------------------------------------------------------
    # 8. Boundary Invariant Rejection (clicks > impressions -> HTTP 422)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    invalid_payload = {
        "brand": "nykaa",
        "impressions": 50000.0,
        "clicks": 60000.0,
    }
    try:
        res = client.post("/forecast_revenue", json=invalid_payload)
        duration = (time.perf_counter() - t0) * 1000.0
        if res.status_code == 422:
            body_text = res.text
            has_invariant_marker = (
                "clicks" in body_text and "impressions" in body_text
            ) or "Logical invariant violated" in body_text
            if has_invariant_marker:
                results.append(
                    StepResult(
                        name="8. Boundary Invariant Rejection (clicks > impressions -> HTTP 422)",
                        passed=True,
                        detail="Correctly rejected with HTTP 422 and logical invariant violation error",
                        duration_ms=duration,
                    )
                )
            else:
                results.append(
                    StepResult(
                        name="8. Boundary Invariant Rejection (clicks > impressions -> HTTP 422)",
                        passed=False,
                        detail=f"HTTP 422 returned but unexpected error detail: {body_text}",
                        duration_ms=duration,
                    )
                )
        else:
            results.append(
                StepResult(
                    name="8. Boundary Invariant Rejection (clicks > impressions -> HTTP 422)",
                    passed=False,
                    detail=f"Expected HTTP 422 for clicks > impressions, got {res.status_code}: {res.text}",
                    duration_ms=duration,
                )
            )
    except Exception as exc:
        results.append(
            StepResult(
                name="8. Boundary Invariant Rejection (clicks > impressions -> HTTP 422)",
                passed=False,
                detail=f"Exception during request: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    return results


def print_results(results: list[StepResult], mode_desc: str) -> bool:
    """Print formatted pass/fail logs and return True if all passed."""
    print("=" * 78)
    print("  MARKETING CAMPAIGN INTELLIGENCE - END-TO-END SMOKE VERIFICATION")
    print("=" * 78)
    print(f" Execution Mode: {mode_desc}")
    print("-" * 78)

    all_passed = True
    for res in results:
        status_tag = "[PASS]" if res.passed else "[FAIL]"
        print(f" {status_tag} {res.name} ({res.duration_ms:.1f}ms)")
        print(f"        Detail: {res.detail}")
        if not res.passed:
            all_passed = False

    print("-" * 78)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    total_time = sum(r.duration_ms for r in results)

    print(
        f" SUMMARY: {passed_count}/{len(results)} passed | "
        f"{failed_count} failed | Total latency: {total_time:.1f}ms"
    )
    if all_passed:
        print(" [SUCCESS] All microservice operational & ML inference contracts verified!")
    else:
        print(" [FAILURE] One or more smoke checks failed.")
    print("=" * 78)
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-End Smoke Verification for Marketing Campaign Intelligence API."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of running FastAPI microservice (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline in-process verification using FastAPI TestClient in lifespan context",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds (default: 10.0)",
    )
    args = parser.parse_args()

    # Determine whether to run via TestClient (--mock) or live server
    use_mock = args.mock
    live_client: httpx.Client | None = None

    if not use_mock:
        # Check if live server is reachable
        try:
            test_resp = httpx.get(f"{args.url.rstrip('/')}/health", timeout=3.0)
            if test_resp.status_code in (200, 503):
                live_client = httpx.Client(base_url=args.url.rstrip("/"), timeout=args.timeout)
            else:
                print(f"[WARN] Live server at {args.url} responded with status {test_resp.status_code}.")
                print("Falling back to offline --mock mode via FastAPI TestClient.")
                use_mock = True
        except (httpx.ConnectError, httpx.TimeoutException, Exception) as exc:
            print(f"[INFO] Live server at {args.url} is unreachable ({type(exc).__name__}).")
            print("Falling back to offline --mock mode via FastAPI TestClient in lifespan context.\n")
            use_mock = True

    if use_mock:
        if TestClient is None or app is None:
            print("[ERROR] Cannot run in --mock mode: FastAPI or app import failed.")
            return 1
        with TestClient(app) as test_client:
            results = run_smoke_tests(test_client)
            passed = print_results(results, mode_desc="In-Process FastAPI TestClient (Lifespan Context)")
    else:
        assert live_client is not None
        with live_client:
            results = run_smoke_tests(live_client)
            passed = print_results(results, mode_desc=f"Live Microservice at {args.url}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
