# Data Setup

## Included in Git (demo / dashboard)

| File | Purpose |
|------|---------|
| `nykaa_campaign_data_sample.csv` | Nykaa brand sample |
| `purplle_campaign_data_sample.csv` | Purplle brand sample |
| `tira_campaign_data_sample.csv` | Tira brand sample |

## Full dataset (local only)

| File | Purpose |
|------|---------|
| `nykaa_campaign_data.csv` | Full Nykaa campaign export |
| `purplle_campaign_data.csv` | Full Purplle campaign export |
| `tira_campaign_data.csv` | Full Tira campaign export |

**How to obtain:** Place your original brand CSV exports in `data/` with the filenames above.

**Resolution order:** `src/data_ingestion.py` prefers full files; falls back to `*_sample.csv`.

**Ingest:** `python -m src.data_ingestion` (requires PostgreSQL; copy `.env.example` to `.env`).
