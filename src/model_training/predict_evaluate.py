# predict_evaluate.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# file paths
MODEL_PATH = "/Users/anuragchaubey/RouteWise/models/best_tuned_model.pkl"
DATA_PATH = "/Users/anuragchaubey/RouteWise/data/processed/encoded_data.csv"
SAVE_PREDICTIONS = True
PREDICTION_OUTPUT = "/Users/anuragchaubey/RouteWise/data/external/model_predictions.csv"

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Encoded data not found at: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["delivery_time"])
    y = df["delivery_time"]
    return X, y

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\nmodel evaluation:")
    print(f"  MAE :  {mae:.2f}")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  R²   :  {r2:.4f}")
    return mae, rmse, r2

def save_predictions(X, y_true, y_pred, output_path):
    df_preds = X.copy()
    df_preds["Actual"] = y_true
    df_preds["Predicted"] = y_pred
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_preds.to_csv(output_path, index=False)
    print(f"\n predictions saved to: {output_path}")

def main():
    print(" loading encoded data and model...")
    X, y = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)

    print(" Predicting delivery time...")
    y_pred = model.predict(X)

    evaluate_predictions(y, y_pred)

    if SAVE_PREDICTIONS:
        save_predictions(X, y, y_pred, PREDICTION_OUTPUT)

if __name__ == "__main__":
    main()
