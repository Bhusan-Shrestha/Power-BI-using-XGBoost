import os
import pandas as pd
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.tables import MonthlySales, Prediction, Product
from ..schemas.sales_schema import PredictionOut, PredictionRequest, PredictionRunResponse
from ..services.notebook_prediction_service import NotebookPredictionService
from ..services.prediction_service import PredictionService

router = APIRouter()


@router.post("/predict", response_model=PredictionRunResponse)
def generate_predictions(payload: PredictionRequest, db: Session = Depends(get_db)):
    rows = db.query(MonthlySales).order_by(MonthlySales.product_id, MonthlySales.date).all()
    if not rows:
        raise HTTPException(status_code=400, detail="No sales data available. Upload data first.")

    prediction_engine = os.getenv("PREDICTION_ENGINE", "db").strip().lower()
    prediction_df = None

    if prediction_engine == "notebook":
        notebook_service = NotebookPredictionService()
        try:
            notebook_df = notebook_service.predict_with_notebook(
                payload.predict_year,
                payload.predict_month,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Notebook execution failed: {exc}") from exc

        required_columns = {"product", "predicted_sales", "predicted_profit"}
        missing_columns = required_columns.difference(notebook_df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction output missing columns: {sorted(missing_columns)}",
            )

        product_map = {
            product.product_name.strip().lower(): product.product_id
            for product in db.query(Product).all()
        }
        notebook_df["product_id"] = notebook_df["product"].astype(str).str.strip().str.lower().map(product_map)
        prediction_df = notebook_df[["product_id", "predicted_sales", "predicted_profit"]].copy()
    else:
        source_df = pd.DataFrame(
            [
                {
                    "product_id": r.product_id,
                    "date": r.date,
                    "sales": r.sales,
                    "profit": r.profit,
                    "marketing_spend": r.marketing_spend,
                    "region": r.region,
                }
                for r in rows
            ]
        )
        service = PredictionService()
        try:
            prediction_df = service.predict(
                source_df=source_df,
                predict_year=payload.predict_year,
                predict_month=payload.predict_month,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"DB model prediction failed: {exc}") from exc

    db.query(Prediction).delete()

    unmatched_products = []
    inserted_count = 0
    target_month = f"{payload.predict_year}-{payload.predict_month:02d}"

    for _, row in prediction_df.iterrows():
        product_id = row.get("product_id")
        if product_id is None:
            unmatched_products.append(str(row.get("product", "unknown")))
            continue

        db.add(
            Prediction(
                product_id=int(product_id),
                month=target_month,
                predicted_sales=float(row["predicted_sales"]),
                predicted_profit=float(row["predicted_profit"]),
            )
        )
        inserted_count += 1

    db.commit()

    detail = f"Predictions generated successfully using '{prediction_engine}' engine"
    if unmatched_products:
        detail += f" (skipped products not found in DB: {sorted(set(unmatched_products))})"

    return {
        "message": detail,
        "count": inserted_count,
        "month": target_month,
    }


@router.get("/predictions", response_model=List[PredictionOut])
def get_predictions(db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.month.asc(), Prediction.product_id.asc()).all()
    return [
        {
            "id": row.id,
            "product_id": row.product_id,
            "month": row.month,
            "predicted_sales": row.predicted_sales,
            "predicted_profit": row.predicted_profit,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]
