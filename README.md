## Quick Start

1. Configure environment variables from `.env.example`.
2. Train the model:
   - `python ml/train_model.py --input data/sample_sales.xlsx`
3. Run backend:
   - `uvicorn backend.app.main:app --reload --port 8000`
4. Run frontend:
   - `cd frontend && npm install && npm run dev`

Detailed setup is in `docs/setup.md`.
