# Adversarial Reviews

Independent adversarial audits conducted by `adversarial-reviewer` at the conclusion of each milestone.

---

## Milestone M1 Review
**Verdict:** APPROVED  
**Date:** 2026-10-05  
**Test Command:** `pytest -q` -> exit code 0 (56 passed)  
**Base Commit:** `c0c4220ea1ede5968d7369dd6fda558928f6f897`

### Critical Defects (must fix; any defect means REJECTED)
None

### Recommendations (non-blocking)
- `src/inference.py:84`: Normalize channel strings case-insensitively in `build_campaign_row` (e.g. `ch.lower() in [str(s).strip().lower() for s in selected_channels]`). `build_batch_campaign_rows` normalizes channels via `.str.strip().str.lower()`, but `build_campaign_row` performs exact title-cased comparison `ch in selected_channels`. Normalizing both ensures identical one-hot channel vectors even when external callers pass lowercase channel lists (e.g., `["instagram", "google"]`).
- `api/main.py:87`: Consider tightening `channels: list[str]` in `CampaignInput` with a field validator or `Literal["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"]` (with case-normalization) so unknown channel strings are validated at the perimeter rather than falling through to zeroed one-hot columns.
- `api/main.py:178, 192`: Update return type hints on `forecast_revenue` and `predict_profitability` from `-> dict` to `-> RevenuePredictionResponse` and `-> ProfitabilityPredictionResponse` to maintain consistent static typing with the batch endpoints.

---

## Milestone M2 Review
**Verdict:** APPROVED  
**Date:** 2026-10-06  
**Test Command:** `pytest -q` -> exit code 0 (104 passed)  
**Base Commit:** `c0c4220ea1ede5968d7369dd6fda558928f6f897`

### Remediated Critical Defects (from initial review)
- Added missing third-party dependencies (`pyyaml`, `sqlalchemy`, `httpx`) to `requirements.txt` to ensure reproducible execution on clean CI runners.
- Added pre-validation empty-string fallback in `src/config.py` preventing service crash when `.env.example` is copied to `.env`.

### Critical Defects (must fix; any defect means REJECTED)
None

### Recommendations (non-blocking)
- `src/config.py:82`: `get_settings()` is cached via `@lru_cache`. For testing dynamic environment variable changes in downstream integration tests, consider recommending or documenting `get_settings.cache_clear()` to ensure cache invalidation when `monkeypatch.setenv` is applied across test boundaries.
- `Dockerfile:58`: The `RUN chown -R appuser:appgroup /app` layer adds a small additional image layer on top of `COPY --chown=appuser:appgroup` directives (lines 52–55). While completely safe and functional, future image size optimization could consolidate ownership assignment during initial file creation.
- `docker-compose.yml:44`: The `depends_on` condition for the optional `db` service uses `required: false`, which is supported in Docker Compose v2.20+. Ensure CI or target deployment runner environments use modern Compose specifications when spinning up the multi-container stack.


