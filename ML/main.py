from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import joblib
import io

app = FastAPI()

sales_model = joblib.load("models/sales_model.pkl")
profit_model = joblib.load("models/profit_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

@app.post("/predict-next-month/")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Product','Place','Date'])

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    df['Month_sin'] = np.sin(2*np.pi*df['Month']/12)
    df['Month_cos'] = np.cos(2*np.pi*df['Month']/12)

    for lag in [1,2,3]:
        df[f'Sales_lag{lag}'] = df.groupby(['Product','Place'])['Sales'].shift(lag)
        df[f'Profit_lag{lag}'] = df.groupby(['Product','Place'])['Profit'].shift(lag)

    df['Sales_roll3'] = df.groupby(['Product','Place'])['Sales'].rolling(3).mean().reset_index(0,drop=True)
    df['Profit_roll3'] = df.groupby(['Product','Place'])['Profit'].rolling(3).mean().reset_index(0,drop=True)

    df = df.dropna()

    latest = df.groupby(['Product','Place']).last().reset_index()

    # Create next month
    next_data = latest.copy()
    next_data['Month'] += 1
    next_data['Year'] += (next_data['Month'] > 12)
    next_data['Month'] = next_data['Month'].apply(lambda x: 1 if x>12 else x)

    next_data['Month_sin'] = np.sin(2*np.pi*next_data['Month']/12)
    next_data['Month_cos'] = np.cos(2*np.pi*next_data['Month']/12)

    # One-hot same as training
    next_data = pd.get_dummies(next_data, columns=['Product','Place'], drop_first=True)

    # Align columns
    for col in feature_cols:
        if col not in next_data.columns:
            next_data[col] = 0

    next_data = next_data[feature_cols]

    sales_pred = sales_model.predict(next_data)
    profit_pred = profit_model.predict(next_data)

    latest['Predicted_Sales'] = sales_pred
    latest['Predicted_Profit'] = profit_pred

    return latest[['Product','Place','Predicted_Sales','Predicted_Profit']].to_dict(orient="records")
