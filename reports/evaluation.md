# Marketing Campaign Analysis — Evaluation Report

## Regression (Revenue Forecast)
- R²: 0.7189
- RMSE: 252360.02

## Classification (Profitability)
- Accuracy: 0.9686
- Weighted F1: 0.9689

## Notes
- Classification uses revenue + acquisition_cost as features after revenue forecast (leakage-aware pipeline).
- Regenerate: `python src/train_models.py && python scripts/export_evaluation.py`