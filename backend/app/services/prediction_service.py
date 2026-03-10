import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from .predict import load_model_and_meta, merge_into_master, predict_next_month


class PredictionService:
    def __init__(self) -> None:
        self.root_dir = Path(__file__).resolve().parents[3]
        self.input_dir = self.root_dir / "input_data"
        self.ml_output_dir = self.root_dir / "ml" / "outputs"
        self.output_dir = self.root_dir / "output_data"
        self.latest_predictions_file = self.output_dir / "latest_predictions.json"
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

        required = [
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
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"File missing columns: {missing}")

        df_raw, was_merged = merge_into_master(frame)
        result = predict_next_month(df_raw, model, meta)

        generated_name = result["output_file"]
        output_path = self.output_dir / generated_name

        # Support both generation locations during transition:
        # - new flow: output_data
        # - older flow: ml/outputs
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
