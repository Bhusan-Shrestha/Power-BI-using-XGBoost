# System Architecture

## Data Pipeline

1. User uploads Excel in React (`UploadData.jsx`).
2. Frontend sends file to FastAPI `/upload` endpoint.
3. FastAPI reads and cleans file with pandas.
4. Cleaned rows are persisted into PostgreSQL (`products`, `monthly_sales`).
5. `/predict` loads XGBoost model from `ml/model.pkl`.
6. Predictions are generated and written to `predictions` table.
7. Frontend reads `/analytics/*` and `/predictions` for charts.
8. Power BI connects to PostgreSQL and visualizes data.

## Service Boundaries

- Frontend: UI + API client
- Backend: ingestion, analytics APIs, prediction orchestration
- ML: model training and offline prediction scripts
- Database: transactional storage for historical and forecasted data
- Power BI: external BI reporting layer
