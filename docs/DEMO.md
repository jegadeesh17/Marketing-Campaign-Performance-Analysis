# Marketing Campaign Intelligence — Demo Script

5-minute walkthrough for interviews. Run from project root after training.

## Prerequisites

```bash
pip install -r requirements.txt
python src/train_models.py
```

## 1. Evaluation metrics (2 min)

```bash
type reports\evaluation.md
```

Point out:
- Revenue regression R² and RMSE on holdout split
- Profit classification accuracy and weighted F1 (not headline accuracy alone)
- `reports/metrics.json` for programmatic access

## 2. EDA artifacts (1 min)

Open `docs/eda/`:
- `roi_distribution.png` — ROI spread across campaigns
- `correlation_heatmap.png` — feature relationships
- `roi_by_campaign_type.png` — per-type performance spread

## 3. FastAPI (1 min)

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/forecast_revenue -H "Content-Type: application/json" -d "{\"brand\":\"nykaa\"}"
curl -X POST http://127.0.0.1:8000/predict_profitability -H "Content-Type: application/json" -d "{\"brand\":\"nykaa\",\"impressions\":80000,\"clicks\":6000}"
```

## 4. Streamlit dashboard (1 min)

```bash
streamlit run app/app.py
```

- Select brand Nykaa, adjust impressions/clicks
- Show revenue forecast and profit/loss badge
- Mention leakage-aware pipeline: classifier uses forecasted revenue, not ground-truth ROI

## Checklist

- [ ] `reports/evaluation.md` shows real numbers from training
- [ ] Three EDA PNGs exist under `docs/eda/`
- [ ] API `/health` returns 200 with model flags true
- [ ] Streamlit loads without model errors
