# Setup Guide

## 1. Environment

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+

Copy `.env.example` to `.env` and adjust values.

## 2. Install Python Dependencies

From project root:

```bash
pip install -r requirements.txt
```

## 3. Database Setup

Create database `sales_ai` and run:

```sql
-- Execute file
\i database/schema.sql
```

## 4. Train ML Model

```bash
python ml/train_model.py --input data/sample_sales.xlsx --output ml/model.pkl
```

## 5. Run FastAPI Backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`

## 6. Run React Frontend

```bash
cd frontend
npm install
npm run dev
```

## 7. Test Pipeline

1. Open frontend Home page.
2. Upload `data/sample_sales.xlsx`.
3. Visit Analytics page for summary data.
4. Visit Predictions page and click `Generate Predictions`.
5. Confirm records in PostgreSQL `predictions` table.
