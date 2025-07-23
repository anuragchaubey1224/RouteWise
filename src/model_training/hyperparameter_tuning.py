# hyperparameter_tuning.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# model save directory
MODEL_DIR = "/Users/anuragchaubey/RouteWise/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# load cleaned encoded data (not scaled)
def load_raw_data(data_path="/Users/anuragchaubey/RouteWise/data/processed/encoded_data.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["delivery_time"])
    y = df["delivery_time"]
    return X, y, X.columns.tolist()

# tune random forest
def tune_random_forest(X_train, y_train, feature_names):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    params = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    search = RandomizedSearchCV(pipeline, params, n_iter=10, cv=5,
                                 scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_

# tune XGBoost
def tune_xgboost(X_train, y_train, feature_names):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0))
    ])
    params = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__subsample': [0.7, 0.8, 1.0]
    }
    search = RandomizedSearchCV(pipeline, params, n_iter=10, cv=5,
                                 scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_

# Save model
def save_model(model, path):
    try:
        joblib.dump(model, path)
        print(f" Best model saved at: {path}")
    except Exception as e:
        print(f" Error saving model: {e}")

# main tuning logic
def run_hyperparameter_tuning(
    data_path="/Users/anuragchaubey/RouteWise/data/processed/encoded_data.csv",
    test_size=0.2,
    random_state=42,
    final_model_name="best_tuned_model.pkl"
):
    print(" Loading data...")
    X, y, feature_names = load_raw_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("\n Tuning Random Forest...")
    rf_model, rf_score = tune_random_forest(X_train, y_train, feature_names)
    print(f"   RF Best CV MAE: {-rf_score:.2f}")

    print("\n Tuning XGBoost...")
    xgb_model, xgb_score = tune_xgboost(X_train, y_train, feature_names)
    print(f" XGB Best CV MAE: {-xgb_score:.2f}")

    # select best model (scores are negative MAE)
    if rf_score > xgb_score:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = xgb_model
        best_name = "XGBoost"
    
    print(f"\n Selected Model: {best_name}")

    # evaluate on test set
    print("\n Evaluating on test set...")
    y_pred = best_model.predict(X_test)
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"  R2:   {r2_score(y_test, y_pred):.4f}")

    # save best model
    final_model_path = os.path.join(MODEL_DIR, final_model_name)
    save_model(best_model, final_model_path)

if __name__ == "__main__":
    run_hyperparameter_tuning()
