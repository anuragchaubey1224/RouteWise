# outlier detection and removal module

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        # You can make thresholds configurable here if needed
        pass

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        X = X.copy()

        # Domain knowledge based filtering
        X = X[(X["agent_age"] >= 18) & (X["agent_age"] <= 50)]
        X = X[(X["agent_rating"] >= 1) & (X["agent_rating"] <= 5)]
        X = X[(X["delivery_speed_km_min"] >= 0.01) & (X["delivery_speed_km_min"] <= 2)]
        X = X[
            (X["order_hour"].between(0, 23)) &
            (X["order_minute"].between(0, 59)) &
            (X["pickup_hour"].between(0, 23)) &
            (X["pickup_minute"].between(0, 59))
        ]

        # IQR-based filtering
        def remove_iqr_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return df[(df[column] >= lower) & (df[column] <= upper)]

        for col in ["haversine_distance_km", "order_to_pickup_min", "delivery_time"]:
            X = remove_iqr_outliers(X, col)

        return X
