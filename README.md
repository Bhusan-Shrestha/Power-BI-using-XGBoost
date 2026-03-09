# Sales AI Platform

AI-powered Sales and Profit Prediction platform with React + FastAPI + PostgreSQL + XGBoost + Power BI.

## Project Structure

```text
sales-ai-platform/
  frontend/
  backend/
  ml/
  database/
  powerbi/
  docs/
  data/
```

## Quick Start

1. Configure environment variables from `.env.example`.
2. Start PostgreSQL and run `database/schema.sql`.
3. Train the model:
   - `python ml/train_model.py --input data/sample_sales.xlsx`
4. Run backend:
   - `uvicorn backend.app.main:app --reload --port 8000`
5. Run frontend:
   - `cd frontend && npm install && npm run dev`

Detailed setup is in `docs/setup.md`.
