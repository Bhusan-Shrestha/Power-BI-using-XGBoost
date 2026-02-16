import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------- CONFIG --------
DATA_PATH = r"D:\Project\Project VI\Business Intelligence\ML\Sample data.xlsx"
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
TARGETS = ["Sales", "Profit"]
CATEGORICAL_COLS = ["Segment", "Country", "Product", "Discount Band", "Month Name"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- FUNCTIONS --------
def load_data(path, sheet_name="Sheet1"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df, categorical_cols, targets):
    category_mappings = {}
    cat_cols = sorted(
        list(set(categorical_cols + df.select_dtypes(include=["object", "category"]).columns.tolist()) - set(targets))
    )
    for col in cat_cols:
        df[col] = df[col].astype("category")
        category_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    print(f"✅ Preprocessed: {len(cat_cols)} categorical features, {len(numeric_cols)} numeric features")
    return df, cat_cols, category_mappings

def feature_engineering(df):
    # Interaction term
    df["Units_Sold_x_Sale_Price"] = df["Units Sold"] * df["Sale Price"]
    return df

def handle_outliers(df, columns, lower_quantile=0.01, upper_quantile=0.99):
    for col in columns:
        low = df[col].quantile(lower_quantile)
        high = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(low, high)
    print(f"✅ Outliers clipped for: {columns}")
    return df

def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def plot_feature_importance(model, feature_names, target_name):
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values("Importance", ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x="Importance", y="Feature", data=imp_df)
    plt.title(f"{target_name} Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_name}_feature_importance.png"))
    plt.close()
    print(f"✅ Feature importance plot saved for {target_name}")

def train_model_cv(X, y, target_name, use_log=False):
    y_transformed = np.log1p(y) if use_log else y

    param_grid = {
        "n_estimators": [300, 400, 500],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=15,
                                scoring='neg_root_mean_squared_error', cv=kf, verbose=0, random_state=42)
    search.fit(X, y_transformed)
    model = search.best_estimator_
    print(f"\n🎯 {target_name} Best Hyperparameters: {search.best_params_}")

    preds = model.predict(X)
    if use_log:
        preds = np.expm1(preds)

    metrics = compute_metrics(y, preds)
    print(f"📊 {target_name} Metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.6f}")
    return model, preds, metrics

def save_model(model, name):
    path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {path}")

# -------- MAIN SCRIPT --------
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df, cat_cols, category_mappings = preprocess_data(df, CATEGORICAL_COLS, TARGETS)
    df = feature_engineering(df)
    numeric_cols = ["Sales", "Profit", "Units Sold", "Sale Price", "Gross Sales", "COGS"]
    df = handle_outliers(df, numeric_cols)

    feature_cols = [
        "Segment", "Country", "Product", "Discount Band",
        "Units Sold", "Manufacturing Price", "Sale Price",
        "Gross Sales", "Discounts", "COGS",
        "Month Number", "Month Name", "Year",
        "Units_Sold_x_Sale_Price"
    ]

    metrics_dict = {}
    # Train models for both targets
    for target in TARGETS:
        X = df[feature_cols]
        y = df[target]
        use_log = y.min() >= 0

        model, preds, metrics = train_model_cv(X, y, target, use_log=use_log)
        save_model(model, target)
        df[f"{target}_Predicted"] = preds
        metrics_dict[target] = metrics
        plot_feature_importance(model, feature_cols, target)

    # -------- MAP CATEGORICAL CODES BACK TO ORIGINAL NAMES --------
    for col, mapping in category_mappings.items():
        df[col] = df[col].map(mapping)

    # -------- EXPORT COMBINED EXCEL --------
    combined_file = os.path.join(OUTPUT_DIR, "Sales_Profit_All_in_One.xlsx")
    df.to_excel(combined_file, index=False)
    print(f"\n✅ Combined Sales & Profit predictions saved: {combined_file}")

    # -------- SCATTER PLOTS Actual vs Predicted with Metrics Table --------
    scatter_file = os.path.join(OUTPUT_DIR, "Actual_vs_Predicted_with_metrics.png")
    plt.figure(figsize=(16, 7))

    sales_metrics_text = f"Sales Metrics:\nRMSE={metrics_dict['Sales']['rmse']:.2f}\nMAE={metrics_dict['Sales']['mae']:.2f}\nR²={metrics_dict['Sales']['r2']:.6f}"
    profit_metrics_text = f"Profit Metrics:\nRMSE={metrics_dict['Profit']['rmse']:.2f}\nMAE={metrics_dict['Profit']['mae']:.2f}\nR²={metrics_dict['Profit']['r2']:.6f}"

    # Sales subplot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x="Sales", y="Sales_Predicted", data=df, color='blue', s=50)
    plt.plot([df["Sales"].min(), df["Sales"].max()],
             [df["Sales"].min(), df["Sales"].max()], 'r--')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Sales: Actual vs Predicted")
    plt.text(0.05, 0.95, sales_metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # Profit subplot
    plt.subplot(1, 2, 2)
    sns.scatterplot(x="Profit", y="Profit_Predicted", data=df, color='green', s=50)
    plt.plot([df["Profit"].min(), df["Profit"].max()],
             [df["Profit"].min(), df["Profit"].max()], 'r--')
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Profit: Actual vs Predicted")
    plt.text(0.05, 0.95, profit_metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(scatter_file)
    plt.close()
    print(f"✅ Scatter plots with metrics saved: {scatter_file}")

    print("\n🎉 All models trained, predictions exported, and plots generated successfully!")
