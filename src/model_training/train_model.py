import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ✅ Update: Model save directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/")
os.makedirs(MODEL_DIR, exist_ok=True)

# 📌 Utility to calculate metrics
def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# 📌 Optional Cross-validation metrics
def get_cv_metrics(model, X, y, cv: int) -> Dict[str, float]:
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    rmse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    return {
        "CV_MAE_Mean": mae_scores.mean(),
        "CV_MAE_Std": mae_scores.std(),
        "CV_RMSE_Mean": rmse_scores.mean(),
        "CV_RMSE_Std": rmse_scores.std(),
        "CV_R2_Mean": r2_scores.mean(),
        "CV_R2_Std": r2_scores.std(),
    }

# 🚀 Main training function
def train_models(
    df: pd.DataFrame,
    target_column: str = "delivery_time",
    test_size: float = 0.2,
    random_state: int = 42,
    do_cross_validation: bool = False,
    cv_folds: int = 5
) -> Dict[str, Dict]:
    """
    Trains multiple regression models and returns performance metrics and trained models.

    Returns:
        {
            'trained_models': {model_name: model_object},
            'performance_metrics': {model_name: {metric_name: value}}
        }
    """
    if target_column not in df.columns:
        raise ValueError(f"❌ Target column '{target_column}' not found in the dataframe.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Models to train
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(random_state=random_state, verbosity=0),
    }

    results = {
        "trained_models": {},
        "performance_metrics": {}
    }

    for model_name, model in models.items():
        print(f"\n✅ Training: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        test_metrics = calculate_metrics(y_test, y_pred)
        results["performance_metrics"][model_name] = test_metrics

        print(f"📊 Test: MAE={test_metrics['MAE']:.2f}, RMSE={test_metrics['RMSE']:.2f}, R2={test_metrics['R2']:.4f}")

        # Cross-validation
        if do_cross_validation:
            print(f"🔁 Performing {cv_folds}-Fold CV for {model_name}...")
            cv_metrics = get_cv_metrics(model, X, y, cv=cv_folds)
            results["performance_metrics"][model_name].update(cv_metrics)

        # ✅ Save the model to RouteWise's model folder
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"💾 Model saved to: {model_path}")

        # Add to trained model dictionary
        results["trained_models"][model_name] = model

    return results
