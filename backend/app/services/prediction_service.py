import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

from .predict import load_model_and_meta, merge_into_master, predict_next_month


class PredictionService:
    def __init__(self) -> None:
        self.root_dir = Path(__file__).resolve().parents[3]
        self.input_dir = self.root_dir / "input_data"
        self.ml_output_dir = self.root_dir / "ml" / "outputs"
        self.master_data_path = self.root_dir / "ml" / "data_sets.xlsx"
        self.output_dir = self.root_dir / "output_data"
        self.latest_predictions_file = self.output_dir / "latest_predictions.json"
        self.required_columns = [
            "Product",
            "Segment",
            "Discount Band",
            "Units Sold",
            "Manufacturing Price",
            "Sale Price",
            "Gross Sales",
            "Discounts",
            "Sales",
            "COGS",
            "Profit",
            "Month Number",
            "Year",
        ]
        self.discount_rates = {
            "None": 0.0,
            "Low": 0.05,
            "Medium": 0.1,
            "High": 0.2,
        }
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.ml_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> dict:
        if not file.filename:
            raise ValueError("Uploaded file name is missing")
        if not file.filename.lower().endswith((".xlsx", ".xls")):
            raise ValueError("Only Excel files are supported")

        destination = self.input_dir / file.filename
        content = await file.read()
        destination.write_bytes(content)

        frame = pd.read_excel(destination)
        products_upserted = int(frame["Product"].nunique()) if "Product" in frame.columns else 0

        return {
            "path": destination,
            "rows": int(len(frame)),
            "products": products_upserted,
        }

    def _latest_input_file(self) -> Path:
        candidates = sorted(
            self.input_dir.glob("*.xls*"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No input file found. Please upload data first.")
        return candidates[0]

    def run_prediction(self) -> dict:
        model, meta = load_model_and_meta()
        input_file = self._latest_input_file()

        frame = pd.read_excel(input_file)
        frame["Discount Band"] = frame["Discount Band"].fillna("None")

        missing = [column for column in self.required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"File missing columns: {missing}")

        df_raw, was_merged = merge_into_master(frame)
        result = predict_next_month(df_raw, model, meta)

        generated_name = result["output_file"]
        output_path = self.output_dir / generated_name

        source_candidates = [
            output_path,
            self.ml_output_dir / generated_name,
        ]
        generated_path = next((path for path in source_candidates if path.exists()), None)
        if generated_path is None:
            raise FileNotFoundError(f"Prediction output not found: {generated_name}")

        if generated_path != output_path:
            shutil.copy2(generated_path, output_path)

        rows = []
        for item in result.get("predictions", []):
            rows.append(
                {
                    "product_id": str(item.get("product", "")),
                    "month": result.get("predicting_month"),
                    "predicted_sales": float(item.get("predicted_sales", 0.0)),
                    "predicted_unit_sales": float(item.get("predicted_units", 0.0)),
                }
            )

        self.latest_predictions_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        result["merged"] = was_merged
        result["input_file"] = input_file.name
        result["download_file"] = generated_name
        result["download_url"] = f"/download/{generated_name}"
        return result

    def read_latest_predictions(self) -> list[dict]:
        if not self.latest_predictions_file.exists():
            return []

        try:
            return json.loads(self.latest_predictions_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _load_master_data(self) -> pd.DataFrame:
        if not self.master_data_path.exists():
            raise FileNotFoundError(
                "Master dataset not found. Please create ml/data_sets.xlsx with Monthly_Data sheet."
            )

        frame = pd.read_excel(self.master_data_path, sheet_name="Monthly_Data")
        if "Discount Band" in frame.columns:
            frame["Discount Band"] = frame["Discount Band"].fillna("None")
        return frame

    def _save_master_data(self, frame: pd.DataFrame) -> None:
        with pd.ExcelWriter(
            self.master_data_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            frame.to_excel(writer, sheet_name="Monthly_Data", index=False)

    def add_sales_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = self._load_master_data()
        missing_columns = [column for column in self.required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Master dataset missing columns: {missing_columns}")

        product = str(payload.get("product", "")).strip()
        segment = str(payload.get("segment", "")).strip()
        discount_band = str(payload.get("discount_band", "None")).strip() or "None"
        if discount_band not in self.discount_rates:
            raise ValueError("Discount Band must be one of: None, Low, Medium, High")

        month_number = int(payload.get("month_number"))
        year = int(payload.get("year"))
        units_sold = float(payload.get("units_sold"))
        manufacturing_price = float(payload.get("manufacturing_price"))
        sale_price = float(payload.get("sale_price"))

        if not product:
            raise ValueError("Product is required")
        if not segment:
            raise ValueError("Segment is required")
        if month_number < 1 or month_number > 12:
            raise ValueError("Month Number must be between 1 and 12")
        if year < 2000 or year > 2100:
            raise ValueError("Year must be between 2000 and 2100")
        if units_sold < 0 or manufacturing_price < 0 or sale_price < 0:
            raise ValueError("Units and prices must be non-negative")

        gross_sales = units_sold * sale_price
        discounts = gross_sales * self.discount_rates[discount_band]
        sales = gross_sales - discounts
        cogs = units_sold * manufacturing_price
        profit = sales - cogs

        new_row = {
            "Product": product,
            "Segment": segment,
            "Discount Band": discount_band,
            "Units Sold": units_sold,
            "Manufacturing Price": manufacturing_price,
            "Sale Price": sale_price,
            "Gross Sales": round(gross_sales, 2),
            "Discounts": round(discounts, 2),
            "Sales": round(sales, 2),
            "COGS": round(cogs, 2),
            "Profit": round(profit, 2),
            "Month Number": month_number,
            "Year": year,
        }

        full_row = {column: None for column in frame.columns}
        for column, value in new_row.items():
            full_row[column] = value

        updated = pd.concat([frame, pd.DataFrame([full_row])], ignore_index=True)
        self._save_master_data(updated)

        return {
            "message": "Sales entry added",
            "inserted": {
                "product": product,
                "segment": segment,
                "discount_band": discount_band,
                "units_sold": units_sold,
                "sales": round(sales, 2),
                "profit": round(profit, 2),
                "month_number": month_number,
                "year": year,
            },
        }

    def get_dashboard_overview(self) -> dict[str, Any]:
        frame = self._load_master_data()
        if frame.empty:
            return {
                "kpis": {},
                "monthly_trend": [],
                "segment_performance": [],
                "discount_performance": [],
                "top_products": [],
                "recent_rows": [],
                "predictions": [],
            }

        numeric_columns = [
            "Units Sold",
            "Gross Sales",
            "Discounts",
            "Sales",
            "COGS",
            "Profit",
            "Month Number",
            "Year",
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

        frame["Month Number"] = frame["Month Number"].clip(lower=1, upper=12)

        frame["month_date"] = pd.to_datetime(
            frame["Year"].astype(int).astype(str)
            + "-"
            + frame["Month Number"].astype(int).astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )
        frame = frame.dropna(subset=["month_date"])

        monthly_summary = (
            frame.groupby(["Year", "Month Number", "month_date"], as_index=False)
            .agg(
                sales=("Sales", "sum"),
                profit=("Profit", "sum"),
                units_sold=("Units Sold", "sum"),
            )
            .sort_values("month_date")
        )

        monthly_trend = [
            {
                "month": row["month_date"].strftime("%b %Y"),
                "sales": round(float(row["sales"]), 2),
                "profit": round(float(row["profit"]), 2),
                "units_sold": round(float(row["units_sold"]), 0),
            }
            for _, row in monthly_summary.tail(12).iterrows()
        ]

        segment_performance = [
            {
                "segment": str(row["Segment"]),
                "sales": round(float(row["sales"]), 2),
                "profit": round(float(row["profit"]), 2),
                "units_sold": round(float(row["units_sold"]), 0),
            }
            for _, row in (
                frame.groupby("Segment", as_index=False)
                .agg(sales=("Sales", "sum"), profit=("Profit", "sum"), units_sold=("Units Sold", "sum"))
                .sort_values("sales", ascending=False)
            ).iterrows()
        ]

        discount_performance = [
            {
                "discount_band": str(row["Discount Band"]),
                "sales": round(float(row["sales"]), 2),
                "profit": round(float(row["profit"]), 2),
            }
            for _, row in (
                frame.groupby("Discount Band", as_index=False)
                .agg(sales=("Sales", "sum"), profit=("Profit", "sum"))
                .sort_values("sales", ascending=False)
            ).iterrows()
        ]

        top_products = [
            {
                "product": str(row["Product"]),
                "sales": round(float(row["sales"]), 2),
                "profit": round(float(row["profit"]), 2),
                "units_sold": round(float(row["units_sold"]), 0),
            }
            for _, row in (
                frame.groupby("Product", as_index=False)
                .agg(sales=("Sales", "sum"), profit=("Profit", "sum"), units_sold=("Units Sold", "sum"))
                .sort_values("sales", ascending=False)
                .head(10)
            ).iterrows()
        ]

        recent_rows_frame = (
            frame.sort_values(["month_date", "Product"], ascending=[False, True])
            .head(20)
            .copy()
        )
        recent_rows = [
            {
                "product": str(row.get("Product", "")),
                "segment": str(row.get("Segment", "")),
                "month": row["month_date"].strftime("%b %Y"),
                "discount_band": str(row.get("Discount Band", "None")),
                "units_sold": round(float(row.get("Units Sold", 0)), 0),
                "sales": round(float(row.get("Sales", 0)), 2),
                "profit": round(float(row.get("Profit", 0)), 2),
            }
            for _, row in recent_rows_frame.iterrows()
        ]

        predictions = self.read_latest_predictions()
        prediction_total = round(
            float(sum(float(item.get("predicted_sales", 0)) for item in predictions)),
            2,
        )

        latest_month = monthly_summary.iloc[-1]["month_date"] if not monthly_summary.empty else None
        total_sales = float(frame["Sales"].sum())
        total_profit = float(frame["Profit"].sum())
        total_units = float(frame["Units Sold"].sum())

        kpis = {
            "total_sales": round(total_sales, 2),
            "total_profit": round(total_profit, 2),
            "total_units": round(total_units, 0),
            "avg_profit_margin": round((total_profit / total_sales * 100) if total_sales else 0, 2),
            "products_count": int(frame["Product"].nunique()) if "Product" in frame.columns else 0,
            "segments_count": int(frame["Segment"].nunique()) if "Segment" in frame.columns else 0,
            "latest_month": latest_month.strftime("%B %Y") if latest_month is not None else "-",
            "predicted_sales_total": prediction_total,
        }

        return {
            "kpis": kpis,
            "monthly_trend": monthly_trend,
            "segment_performance": segment_performance,
            "discount_performance": discount_performance,
            "top_products": top_products,
            "recent_rows": recent_rows,
            "predictions": predictions,
        }
