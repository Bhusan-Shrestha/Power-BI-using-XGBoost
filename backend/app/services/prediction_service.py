import os
from pathlib import Path
import joblib
import pandas as pd


class PredictionService:
    def __init__(self) -> None:
        model_path = os.getenv("MODEL_PATH", "ml/model.pkl")
        self.model_path = Path(model_path)

    def _load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        return joblib.load(self.model_path)

    def predict(
        self,
        source_df: pd.DataFrame,
        predict_year: int | None = None,
        predict_month: int | None = None,
    ) -> pd.DataFrame:
        artifacts = self._load_artifacts()
        model = artifacts["model"]
        region_encoder = artifacts["region_encoder"]

        history = source_df.copy()
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values(["product_id", "date"])

        # Predict for requested year/month; defaults to one month ahead.
        if predict_year is not None and predict_month is not None:
            if not 1 <= int(predict_month) <= 12:
                raise ValueError("predict_month must be between 1 and 12")
            target_date = pd.Timestamp(f"{int(predict_year)}-{int(predict_month):02d}-01")
        else:
            target_date = history["date"].max() + pd.offsets.MonthBegin(1)

        latest = history.groupby("product_id", as_index=False).tail(1).copy()
        latest["future_date"] = target_date
        latest["month"] = latest["future_date"].dt.month
        latest["previous_sales"] = latest["sales"]
        latest["region"] = latest["region"].astype(str).str.title()

        unknown_regions = sorted(set(latest["region"]) - set(region_encoder.keys()))
        if unknown_regions:
            raise ValueError(f"Unknown region values for model: {unknown_regions}")

        latest["region_encoded"] = latest["region"].map(region_encoder)
        x = latest[["month", "marketing_spend", "region_encoded", "previous_sales"]]
        pred = model.predict(x)

        output = latest[["product_id", "future_date"]].copy()
        output["predicted_sales"] = pred[:, 0]
        output["predicted_profit"] = pred[:, 1]
        output["month"] = pd.to_datetime(output["future_date"]).dt.to_period("M").astype(str)
        return output[["product_id", "month", "predicted_sales", "predicted_profit"]]
