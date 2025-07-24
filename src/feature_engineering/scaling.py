# scaling.py (scaling module for numerical columns)

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CustomScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        # ensure all column names are lowercase
        X.columns = X.columns.str.lower()

        # define column groups
        std_cols = [
            'agent_age', 'order_hour', 'pickup_hour',
            'haversine_distance_km', 'order_to_pickup_min',
            'delivery_speed_km_min'
        ]
        mm_cols = ['agent_rating', 'order_minute', 'pickup_minute']
        cyc_cols = ['order_dayofweek_sin', 'order_dayofweek_cos']

        # filter only those columns present in the data
        self.standard_scale_cols = [col for col in std_cols + cyc_cols if col in X.columns]
        self.minmax_scale_cols = [col for col in mm_cols if col in X.columns]
        all_scaled = set(self.standard_scale_cols + self.minmax_scale_cols)
        self.passthrough_cols = [col for col in X.columns if col not in all_scaled]

        # create ColumnTransformer
        self.scaler = ColumnTransformer(
            transformers=[
                ('std_scaler', StandardScaler(), self.standard_scale_cols),
                ('minmax_scaler', MinMaxScaler(), self.minmax_scale_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        self.scaler.fit(X)
        self.feature_names_out_ = self.scaler.get_feature_names_out()
        return self

    def transform(self, X):
        X.columns = X.columns.str.lower()
        arr = self.scaler.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)

def get_scaling_pipeline():
    return Pipeline(steps=[
        ('scaling', CustomScalerTransformer())
    ])
