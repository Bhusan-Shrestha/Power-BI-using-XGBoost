import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
DATA_PATH = "monthly_sales.xlsx"
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"
TARGETS = ["Sales", "Profit"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD =================
df = pd.read_excel(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Product', 'Place', 'Date'])

# ================= FEATURE ENGINEERING =================

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Seasonality (cyclic)
df['Month_sin'] = np.sin(2*np.pi*df['Month']/12)
df['Month_cos'] = np.cos(2*np.pi*df['Month']/12)

# Lag features
for lag in [1,2,3]:
    df[f'Sales_lag{lag}'] = df.groupby(['Product','Place'])['Sales'].shift(lag)
    df[f'Profit_lag{lag}'] = df.groupby(['Product','Place'])['Profit'].shift(lag)

# Rolling mean
df['Sales_roll3'] = df.groupby(['Product','Place'])['Sales'].rolling(3).mean().reset_index(0,drop=True)
df['Profit_roll3'] = df.groupby(['Product','Place'])['Profit'].rolling(3).mean().reset_index(0,drop=True)

df = df.dropna()

# One-hot encoding (safe categorical handling)
df = pd.get_dummies(df, columns=['Product','Place'], drop_first=True)

# ================= FEATURES =================
feature_cols = [col for col in df.columns if col not in ['Date','Sales','Profit']]

# ================= TRAIN FUNCTION =================

def train_target(target):

    X = df[feature_cols]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "n_estimators": [300,400,500],
        "max_depth": [4,5,6],
        "learning_rate": [0.01,0.05,0.1],
        "subsample": [0.7,0.8,0.9],
        "colsample_bytree": [0.7,0.8,0.9]
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=15,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X, y)

    best_model = search.best_estimator_

    preds = best_model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"\n🎯 {target} Metrics")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 : {r2:.4f}")

    # Feature importance plot
    importance = best_model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x="Importance", y="Feature", data=imp_df.head(15))
    plt.title(f"{target} Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{target}_importance.png")
    plt.close()

    # Save model
    joblib.dump(best_model, f"{MODEL_DIR}/{target.lower()}_model.pkl")

    return best_model

# ================= TRAIN =================

for target in TARGETS:
    train_target(target)

# Save feature columns (VERY IMPORTANT for API)
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_columns.pkl")

print("\n🔥 Elite forecasting model training complete.")
