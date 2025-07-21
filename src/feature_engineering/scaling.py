# scaling.py 

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list.")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_select = [col for col in self.columns if col in X.columns]
        if not cols_to_select:
            print(f"Warning: None of the specified columns {self.columns} found in data.")
            return pd.DataFrame(index=X.index, columns=[])
        return X[cols_to_select].copy()


class CustomScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        std_cols = [
            'agent_age', 'order_hour', 'pickup_hour',
            'haversine_distance_km', 'order_to_pickup_min',
            'delivery_speed_km_min'
        ]
        mm_cols = ['agent_rating', 'order_minute', 'pickup_minute']
        cyc_cols = ['order_dayofweek_sin', 'order_dayofweek_cos']

        self.standard_scale_cols = [col for col in std_cols + cyc_cols if col in X.columns]
        self.minmax_scale_cols = [col for col in mm_cols if col in X.columns]
        all_scaled = set(self.standard_scale_cols + self.minmax_scale_cols)
        self.passthrough_cols = [col for col in X.columns if col not in all_scaled]

        self.scaler = ColumnTransformer(
            transformers=[
                ('std_scaler', StandardScaler(), self.standard_scale_cols),
                ('minmax_scaler', MinMaxScaler(), self.minmax_scale_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        self.scaler.fit(X)
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = self.scaler.get_feature_names_out()

        return self

    def transform(self, X):
        arr = self.scaler.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)


def get_scaling_pipeline():
    return Pipeline(steps=[
        ('scaling', CustomScalerTransformer())
    ])


# Optional: Script mode if needed later
if __name__ == "__main__":
    input_path = "/Users/anuragchaubey/RouteWise/data/processed/delivery_data_encoded.csv"
    output_path = "/Users/anuragchaubey/RouteWise/data/processed/delivery_data_scaled.csv"

    try:
        df = pd.read_csv(input_path)
        df.columns = df.columns.str.lower()
        scaling_pipeline = get_scaling_pipeline()
        df_scaled = scaling_pipeline.fit_transform(df)
        df_scaled.to_csv(output_path, index=False)
        print(f"✔ Scaling complete and saved to: {output_path}")
    except Exception as e:
        print(f"✖ Error during scaling: {e}")
