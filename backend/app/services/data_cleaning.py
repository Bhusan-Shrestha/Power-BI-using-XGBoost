import pandas as pd

REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "date",
    "sales",
    "profit",
    "marketing_spend",
    "region",
}


def clean_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize headers for flexible incoming files.
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"]).dt.date
    numeric_cols = ["sales", "profit", "marketing_spend", "product_id"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["product_id", "date", "sales", "profit", "marketing_spend", "region"])
    cleaned["product_id"] = cleaned["product_id"].astype(int)
    cleaned["region"] = cleaned["region"].astype(str).str.strip().str.title()
    cleaned["product_name"] = cleaned["product_name"].astype(str).str.strip()
    cleaned["category"] = cleaned["category"].astype(str).str.strip()
    return cleaned


def build_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["month"] = pd.to_datetime(features["date"]).dt.month
    features = features.sort_values(["product_id", "date"])
    features["previous_sales"] = features.groupby("product_id")["sales"].shift(1)
    features["previous_sales"] = features["previous_sales"].fillna(features["sales"])
    return features[["product_id", "date", "month", "marketing_spend", "region", "previous_sales"]]
