from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import pandas as pd
import joblib

from core.database import get_db, AsyncSessionLocal

router = APIRouter()

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
MODEL_SAVE_DIR = BACKEND_ROOT / "ml" / "models" / "saved"
MODEL_FILE = MODEL_SAVE_DIR / "xgboost_sales.pkl"
METRICS_FILE = PROJECT_ROOT / "notebooks" / "outputs" / "model_validation" / "evaluation_metrics.json"
DEFAULT_PERIODS = 30
MODEL_NAME = "xgboost"
METRIC = "sales_units_by_product"
FEATURES = [
    "product_id",
    "dow",
    "month",
    "quarter",
    "doy",
    "year",
    "weekofyear",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "lag_60",
    "roll_7",
    "roll_14",
    "roll_30",
    "roll_60",
    "roll_std_7",
    "roll_std_30",
    "ewm_14",
    "ewm_30",
]


async def _ensure_forecast_table(db: AsyncSession) -> None:
    await db.execute(text("""
        CREATE TABLE IF NOT EXISTS monthly_sales_forecasts (
            id SERIAL PRIMARY KEY,
            forecast_month DATE NOT NULL UNIQUE,
            metric VARCHAR(50) NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            periods INTEGER NOT NULL,
            generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            predictions JSONB NOT NULL,
            summary JSONB NOT NULL,
            metrics JSONB,
            trained_rows INTEGER NOT NULL DEFAULT 0
        )
    """))


def _next_month_start() -> datetime.date:
    now = datetime.now(timezone.utc).date()
    if now.month == 12:
        return now.replace(year=now.year + 1, month=1, day=1)
    return now.replace(month=now.month + 1, day=1)


def _next_month_dates(start_date: datetime.date) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(start_date), periods=DEFAULT_PERIODS, freq="D")


async def _sales_history_from_db(db: AsyncSession) -> pd.DataFrame:
    rows = (await db.execute(text("""
        SELECT
            p.id AS product_id,
            p.name AS product_name,
            DATE(s.sale_date) AS ds,
            COALESCE(SUM(s.quantity), 0)::float AS y
        FROM sales s
        JOIN products p ON p.id = s.product_id
        GROUP BY p.id, p.name, DATE(s.sale_date)
        ORDER BY p.id ASC, ds ASC
    """))).fetchall()

    if not rows:
        return pd.DataFrame(columns=["product_id", "product_name", "ds", "y"])

    df = pd.DataFrame([
        {
            "product_id": int(r.product_id),
            "product_name": r.product_name,
            "ds": r.ds,
            "y": float(r.y or 0),
        }
        for r in rows
    ])
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def _build_complete_daily_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    frames = []
    for pid, g in df.groupby("product_id"):
        ordered = g.sort_values("ds")
        idx = pd.date_range(ordered["ds"].min(), ordered["ds"].max(), freq="D")
        expanded = (
            ordered.set_index("ds")
            .reindex(idx)
            .rename_axis("ds")
            .reset_index()
        )
        expanded["y"] = expanded["y"].fillna(0.0).astype(float)
        expanded["product_id"] = int(pid)
        expanded["product_name"] = (
            expanded["product_name"].ffill().bfill().iloc[0]
            if "product_name" in expanded.columns and expanded["product_name"].notna().any()
            else f"Product {int(pid)}"
        )
        frames.append(expanded[["product_id", "product_name", "ds", "y"]])

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["product_id", "ds"])
        .reset_index(drop=True)
    )


def _load_regression_model() -> Any:
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Trained model file not found at backend/ml/models/saved/xgboost_sales.pkl. "
                "Run the notebook training first to generate model artifacts."
            ),
        )
    return joblib.load(MODEL_FILE)


def _load_training_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}

    try:
        with METRICS_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _predict_next_month_units_by_product(
    model: Any,
    raw_df: pd.DataFrame,
    forecast_start: datetime.date,
) -> list[dict]:
    future_days = _next_month_dates(forecast_start)

    products = (
        raw_df[["product_id", "product_name"]]
        .drop_duplicates()
        .sort_values("product_id")
    )

    histories = {}
    for pid, group in raw_df.groupby("product_id"):
        ordered = group.sort_values("ds")
        histories[int(pid)] = [float(v) for v in ordered["y"].tolist()]

    agg = {}
    for _, row in products.iterrows():
        pid = int(row["product_id"])
        agg[pid] = {
            "product_id": pid,
            "product_name": str(row["product_name"]),
            "predicted_units": 0.0,
        }

    for day in future_days:
        for pid, info in agg.items():
            hist = histories.get(pid, [])
            if len(hist) == 0:
                continue

            lag1 = hist[-1]
            lag7 = hist[-7] if len(hist) >= 7 else hist[-1]
            lag14 = hist[-14] if len(hist) >= 14 else hist[-1]
            lag30 = hist[-30] if len(hist) >= 30 else hist[-1]
            lag60 = hist[-60] if len(hist) >= 60 else hist[-1]
            roll7 = float(np.mean(hist[-7:])) if len(hist) >= 7 else float(np.mean(hist))
            roll14 = float(np.mean(hist[-14:])) if len(hist) >= 14 else float(np.mean(hist))
            roll30 = float(np.mean(hist[-30:])) if len(hist) >= 30 else float(np.mean(hist))
            roll60 = float(np.mean(hist[-60:])) if len(hist) >= 60 else float(np.mean(hist))
            roll_std_7 = float(np.std(hist[-7:], ddof=1)) if len(hist) >= 8 else float(np.std(hist))
            roll_std_30 = float(np.std(hist[-30:], ddof=1)) if len(hist) >= 31 else float(np.std(hist))
            ewm_14 = float(pd.Series(hist).ewm(span=14, adjust=False).mean().iloc[-1])
            ewm_30 = float(pd.Series(hist).ewm(span=30, adjust=False).mean().iloc[-1])

            row = pd.DataFrame([
                {
                    "product_id": pid,
                    "dow": day.dayofweek,
                    "month": day.month,
                    "quarter": ((day.month - 1) // 3) + 1,
                    "doy": day.dayofyear,
                    "year": day.year,
                    "weekofyear": int(day.isocalendar().week),
                    "is_weekend": int(day.dayofweek >= 5),
                    "month_sin": float(np.sin(2.0 * np.pi * day.month / 12.0)),
                    "month_cos": float(np.cos(2.0 * np.pi * day.month / 12.0)),
                    "dow_sin": float(np.sin(2.0 * np.pi * day.dayofweek / 7.0)),
                    "dow_cos": float(np.cos(2.0 * np.pi * day.dayofweek / 7.0)),
                    "lag_1": lag1,
                    "lag_7": lag7,
                    "lag_14": lag14,
                    "lag_30": lag30,
                    "lag_60": lag60,
                    "roll_7": roll7,
                    "roll_14": roll14,
                    "roll_30": roll30,
                    "roll_60": roll60,
                    "roll_std_7": roll_std_7,
                    "roll_std_30": roll_std_30,
                    "ewm_14": ewm_14,
                    "ewm_30": ewm_30,
                }
            ])

            row = row[FEATURES]
            pred = float(model.predict(row)[0])
            pred = max(pred, 0.0)
            info["predicted_units"] += pred
            hist.append(pred)
            histories[pid] = hist

    out = []
    for _, item in agg.items():
        total = float(item["predicted_units"])
        out.append({
            "product_id": item["product_id"],
            "product_name": item["product_name"],
            "predicted_units": round(total, 2),
            "lower_units": round(max(total * 0.88, 0.0), 2),
            "upper_units": round(total * 1.12, 2),
        })

    out.sort(key=lambda x: float(x["predicted_units"]), reverse=True)
    for i, item in enumerate(out, start=1):
        item["rank"] = i
    return out


def _build_summary(predictions: list[dict]) -> dict:
    if not predictions:
        return {
            "total_predicted_units": 0,
            "product_count": 0,
            "top_product": "-",
            "top_predicted_units": 0,
        }

    vals = [float(p["predicted_units"]) for p in predictions]
    top = max(predictions, key=lambda p: float(p["predicted_units"]))
    return {
        "total_predicted_units": round(float(np.sum(vals)), 2),
        "product_count": len(predictions),
        "top_product": top.get("product_name", "-"),
        "top_predicted_units": round(float(top.get("predicted_units", 0)), 2),
    }


async def _generate_sales_forecast(db: AsyncSession, periods: int = DEFAULT_PERIODS) -> dict:
    raw_df = await _sales_history_from_db(db)
    if len(raw_df) < 1:
        raise HTTPException(
            status_code=400,
            detail="Not enough sales history to generate monthly forecast.",
        )

    df = _build_complete_daily_history(raw_df)
    model = _load_regression_model()
    metrics = _load_training_metrics()
    forecast_month = _next_month_start()
    predictions = _predict_next_month_units_by_product(model, df, forecast_month)

    if not predictions:
        raise HTTPException(
            status_code=400,
            detail="Not enough product-level history to generate monthly forecast.",
        )

    summary = _build_summary(predictions)

    return {
        "metric": METRIC,
        "model": MODEL_NAME,
        "periods": periods,
        "predictions": predictions,
        "summary": summary,
        "metrics": metrics,
        "trained_rows": int(len(df)),
        "model_file": str(MODEL_FILE),
    }


async def _ensure_monthly_snapshot(db: AsyncSession, force_regenerate: bool = False) -> dict:
    await _ensure_forecast_table(db)
    forecast_month = _next_month_start()

    existing = (await db.execute(text("""
        SELECT forecast_month, metric, model_name, periods, generated_at,
               predictions, summary, metrics, trained_rows
        FROM monthly_sales_forecasts
        WHERE forecast_month = :forecast_month
        LIMIT 1
    """), {"forecast_month": forecast_month})).fetchone()

    if existing and not force_regenerate:
        return {
            "forecast_month": str(existing.forecast_month),
            "metric": existing.metric,
            "model": existing.model_name,
            "periods": int(existing.periods),
            "generated_at": existing.generated_at.isoformat() if existing.generated_at else None,
            "predictions": existing.predictions or [],
            "summary": existing.summary or {},
            "metrics": existing.metrics or {},
            "trained_rows": int(existing.trained_rows or 0),
            "source": "database",
        }

    generated = await _generate_sales_forecast(db, DEFAULT_PERIODS)

    await db.execute(text("""
        INSERT INTO monthly_sales_forecasts (
            forecast_month, metric, model_name, periods,
            predictions, summary, metrics, trained_rows
        ) VALUES (
            :forecast_month, :metric, :model_name, :periods,
            CAST(:predictions AS JSONB), CAST(:summary AS JSONB),
            CAST(:metrics AS JSONB), :trained_rows
        )
        ON CONFLICT (forecast_month)
        DO UPDATE SET
            predictions = EXCLUDED.predictions,
            summary = EXCLUDED.summary,
            metrics = EXCLUDED.metrics,
            trained_rows = EXCLUDED.trained_rows,
            generated_at = NOW()
    """), {
        "forecast_month": forecast_month,
        "metric": generated["metric"],
        "model_name": generated["model"],
        "periods": generated["periods"],
        "predictions": json.dumps(generated["predictions"]),
        "summary": json.dumps(generated["summary"]),
        "metrics": json.dumps(generated["metrics"]),
        "trained_rows": generated["trained_rows"],
    })

    return {
        "forecast_month": str(forecast_month),
        "metric": generated["metric"],
        "model": generated["model"],
        "periods": generated["periods"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": generated["predictions"],
        "summary": generated["summary"],
        "metrics": generated["metrics"],
        "trained_rows": generated["trained_rows"],
        "source": "generated",
    }


async def _delete_latest_snapshot(db: AsyncSession) -> str | None:
    await _ensure_forecast_table(db)
    deleted = (await db.execute(text("""
        DELETE FROM monthly_sales_forecasts
        WHERE id IN (
            SELECT id
            FROM monthly_sales_forecasts
            ORDER BY forecast_month DESC, generated_at DESC
            LIMIT 1
        )
        RETURNING forecast_month
    """))).fetchone()
    if not deleted:
        return None
    return str(deleted.forecast_month)


@router.get("/latest")
async def latest_sales_forecast(db: AsyncSession = Depends(get_db)):
    return await _ensure_monthly_snapshot(db)


@router.post("/refresh-monthly")
async def refresh_monthly_sales_forecast(db: AsyncSession = Depends(get_db)):
    return await _ensure_monthly_snapshot(db, force_regenerate=True)


@router.post("/rebuild-next-month")
async def rebuild_next_month_sales_forecast(db: AsyncSession = Depends(get_db)):
    deleted_month = await _delete_latest_snapshot(db)
    rebuilt = await _ensure_monthly_snapshot(db, force_regenerate=True)
    return {
        "deleted_forecast_month": deleted_month,
        "rebuilt": rebuilt,
    }


async def ensure_monthly_forecast_snapshot() -> None:
    async with AsyncSessionLocal() as db:
        await _ensure_monthly_snapshot(db)
        await db.commit()


