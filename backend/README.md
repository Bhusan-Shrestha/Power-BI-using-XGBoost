# Backend (FastAPI)

## Run

From project root:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

## Endpoints

- `POST /upload` - Upload Excel file and ingest data.
- `GET /sales` - Fetch stored monthly sales rows.
- `POST /predict` - Generate predictions for user-selected month/year from uploaded DB data.
- `GET /predictions` - Fetch predicted rows.
- `GET /analytics/summary` - Total sales, total profit, margin.
- `GET /analytics/regional` - Region-level aggregates.

### `POST /predict` request body

```json
{
	"predict_year": 2026,
	"predict_month": 3
}
```

## Notes

- Database connection uses `DATABASE_URL` from `.env`.
- Model file path uses `MODEL_PATH` from `.env`.
- Upload directory uses `UPLOAD_DIR` from `.env`.
- `PREDICTION_ENGINE=db` uses PostgreSQL uploaded data + `ml/model.pkl` (recommended).
- `PREDICTION_ENGINE=notebook` runs `ml/predict.ipynb` and maps notebook output into DB.
- Notebook path can be configured with `NOTEBOOK_PREDICT_PATH` (default: `ml/predict.ipynb`).
