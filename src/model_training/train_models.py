# train_models.py

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

#  model save directory
MODEL_DIR = "/Users/anuragchaubey/RouteWise/models"
os.makedirs(MODEL_DIR, exist_ok=True)

#  Metric calculator
def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

#  cross-validation metric calculator
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

# main training function
def train_models(
    df: pd.DataFrame,
    target_column: str = "delivery_time",
    test_size: float = 0.2,
    random_state: int = 42,
    do_cross_validation: bool = False,
    cv_folds: int = 5
) -> Dict[str, Dict]:

    if target_column not in df.columns:
        raise ValueError(f" Target column '{target_column}' not found in the dataframe.")

    #  split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    #  train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    #  define models
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
        print(f"\n Training: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        #  evaluate performance
        test_metrics = calculate_metrics(y_test, y_pred)
        results["performance_metrics"][model_name] = test_metrics

        print(f"📊 Test: MAE={test_metrics['MAE']:.2f}, RMSE={test_metrics['RMSE']:.2f}, R2={test_metrics['R2']:.4f}")

        #  cross-validation (optional)
        if do_cross_validation:
            print(f" Performing {cv_folds}-Fold CV for {model_name}...")
            cv_metrics = get_cv_metrics(model, X, y, cv=cv_folds)
            results["performance_metrics"][model_name].update(cv_metrics)

            print(f"   CV Mean: MAE={cv_metrics['CV_MAE_Mean']:.2f}, RMSE={cv_metrics['CV_RMSE_Mean']:.2f}, R2={cv_metrics['CV_R2_Mean']:.4f}")
            print(f"   CV Std : MAE={cv_metrics['CV_MAE_Std']:.2f}, RMSE={cv_metrics['CV_RMSE_Std']:.2f}, R2={cv_metrics['CV_R2_Std']:.4f}")

        # save model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        try:
            joblib.dump(model, model_path)
            print(f" Model saved to: {model_path}")
        except Exception as e:
            print(f" Error saving {model_name}: {e}")

        results["trained_models"][model_name] = model

    return results
