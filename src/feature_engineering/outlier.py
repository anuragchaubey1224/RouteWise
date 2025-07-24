# outlier.py (full outlier removal module)

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # no training required

    def transform(self, X):
        X = X.copy()

        # domain based filtering
        if "agent_age" in X.columns:
            X = X[(X["agent_age"] >= 18) & (X["agent_age"] <= 50)]
        if "agent_rating" in X.columns:
            X = X[(X["agent_rating"] >= 1) & (X["agent_rating"] <= 5)]

        for col in ["order_hour", "order_minute", "pickup_hour", "pickup_minute"]:
            if col in X.columns:
                if "hour" in col:
                    X = X[X[col].between(0, 23)]
                else:
                    X = X[X[col].between(0, 59)]

        #  IQR-based filtering
        def remove_iqr_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return df[(df[column] >= lower) & (df[column] <= upper)]

        for col in ["haversine_distance_km", "order_to_pickup_min"]:
            if col in X.columns:
                X = remove_iqr_outliers(X, col)

        return X
