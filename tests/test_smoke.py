"""Tests for end-to-end smoke verification script (scripts/smoke_test.py)."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.main import app
from scripts.smoke_test import print_results, run_smoke_tests


def test_smoke_verification_in_process_lifespan():
    """Verify run_smoke_tests executes all 8 checks successfully against TestClient in lifespan."""
    with TestClient(app) as client:
        results = run_smoke_tests(client)
        assert len(results) == 8
        failed = [r for r in results if not r.passed]
        assert not failed, f"Smoke tests failed: {[f.name + ': ' + f.detail for f in failed]}"
        assert all(r.passed for r in results)


def test_smoke_cli_mock_flag():
    """Verify that 'python scripts/smoke_test.py --mock' exits 0 with full pass summary."""
    script_path = os.path.join(ROOT, "scripts", "smoke_test.py")
    result = subprocess.run(
        [sys.executable, script_path, "--mock"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Script failed with code {result.returncode}:\n{result.stderr}\n{result.stdout}"
    assert "8/8 passed" in result.stdout
    assert "[SUCCESS]" in result.stdout
    assert "[PASS] 1. Operational Health Probe" in result.stdout
    assert "[PASS] 2. Readiness Probe" in result.stdout
    assert "[PASS] 3. Dark-Mode Glassmorphic Web App" in result.stdout
    assert "[PASS] 4. Single Revenue Forecasting" in result.stdout
    assert "[PASS] 5. Single Profitability Prediction" in result.stdout
    assert "[PASS] 6. Batch Revenue Forecasting" in result.stdout
    assert "[PASS] 7. Batch Profitability Prediction" in result.stdout
    assert "[PASS] 8. Boundary Invariant Rejection" in result.stdout


def test_smoke_verification_failure_reporting():
    """Verify that run_smoke_tests handles client failures gracefully without raising exceptions."""
    failing_client = MagicMock()
    # Mock /health to return 500 error
    error_response = MagicMock()
    error_response.status_code = 500
    error_response.text = "Internal Server Error"
    failing_client.get.return_value = error_response
    failing_client.post.return_value = error_response

    results = run_smoke_tests(failing_client)
    assert len(results) == 8
    health_result = results[0]
    assert health_result.name.startswith("1. Operational Health Probe")
    assert health_result.passed is False
    assert "Expected HTTP 200, got 500" in health_result.detail

    passed = print_results(results, mode_desc="Failing Mock Test")
    assert passed is False


def test_smoke_verification_malformed_payload_handling():
    """Verify that run_smoke_tests flags malformed/invalid payload data even if HTTP 200 is returned."""
    malformed_client = MagicMock()
    # Returns 200 but with empty or invalid JSON structure
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {"status": "unrecognized_status", "unexpected": 123}
    mock_resp.text = '{"status": "unrecognized_status"}'
    malformed_client.get.return_value = mock_resp
    malformed_client.post.return_value = mock_resp

    results = run_smoke_tests(malformed_client)
    assert len(results) == 8
    # All steps should fail validation because the payload structure does not match expectations
    for r in results:
        assert r.passed is False, f"Expected step {r.name} to fail on malformed data, but passed"


def test_smoke_cli_invalid_argument():
    """Verify that smoke_test.py exits with code 2 on invalid CLI options (invalid input)."""
    script_path = os.path.join(ROOT, "scripts", "smoke_test.py")
    result = subprocess.run(
        [sys.executable, script_path, "--unrecognized-option-xyz"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr.lower() or "error" in result.stderr.lower()


def test_demo_documentation_alignment():
    """Verify docs/DEMO.md matches acceptance criteria for M3-TASK-03.

    - Covers single and batch prediction curl examples
    - Dark-mode glassmorphic web app (/app) walkthrough
    - Probe inspection steps (/health and /ready)
    - Single-container port 8000 deployment commands
    - No active Streamlit run commands or port 8501 references
    """
    demo_path = os.path.join(ROOT, "docs", "DEMO.md")
    assert os.path.exists(demo_path), "docs/DEMO.md must exist"

    with open(demo_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify key endpoints are documented with curl or HTTP references
    assert "/health" in content
    assert "/ready" in content
    assert "/forecast_revenue" in content
    assert "/predict_profitability" in content
    assert "/forecast_revenue/batch" in content
    assert "/predict_profitability/batch" in content
    assert "/app" in content

    # Verify single port 8000 and container deployment
    assert "8000" in content
    assert "docker compose up --build" in content or "docker compose" in content

    # Verify no active Streamlit run commands or port 8501 access URLs
    import re
    assert not re.search(r"\bstreamlit\s+run\b", content, re.IGNORECASE)
    assert "http://127.0.0.1:8501" not in content
    assert "localhost:8501" not in content
    assert ":8501" not in content.replace("port 8501", "")



