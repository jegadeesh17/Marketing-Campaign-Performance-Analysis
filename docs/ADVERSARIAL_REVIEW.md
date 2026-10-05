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

