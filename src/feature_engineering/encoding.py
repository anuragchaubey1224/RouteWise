# encoding.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class CyclicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col=None, max_val=6):
        self.col = col
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        radians = 2 * np.pi * X[self.col] / (self.max_val + 1)
        X[f'{self.col}_sin'] = np.sin(radians)
        X[f'{self.col}_cos'] = np.cos(radians)
        X.drop(columns=[self.col], inplace=True)
        return X


class EncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ordinal_cols = ['traffic']
        self.onehot_cols = ['weather', 'vehicle', 'area', 'category']
        self.cyclic_col = 'order_dayofweek'
        self.encoder = None
        self.encoded_columns = None
        self.cyclic_encoder = CyclicEncoder(col=self.cyclic_col)

    def fit(self, X, y=None):
        X = X.copy()

        # Step 1: Apply cyclic encoding
        X = self.cyclic_encoder.transform(X)

        # Step 2: Set up encoders
        ordinal_encoder = OrdinalEncoder(
            categories=[['Low', 'Medium', 'High', 'Jam']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )

        onehot_encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )

        self.encoder = ColumnTransformer(
            transformers=[
                ('ord', ordinal_encoder, self.ordinal_cols),
                ('ohe', onehot_encoder, self.onehot_cols)
            ],
            remainder='passthrough'
        )

        self.encoder.fit(X)

        # Column name extraction
        ordinal_names = self.ordinal_cols
        onehot_names = self.encoder.named_transformers_['ohe'].get_feature_names_out(self.onehot_cols)
        
        passthrough_start = len(self.ordinal_cols) + len(onehot_names)
        passthrough_cols = [
            col for col in X.columns
            if col not in self.ordinal_cols + self.onehot_cols
        ]

        self.encoded_columns = ordinal_names + list(onehot_names) + passthrough_cols
        return self

    def transform(self, X):
        X = X.copy()
        X = self.cyclic_encoder.transform(X)
        transformed = self.encoder.transform(X)
        return pd.DataFrame(transformed, columns=self.encoded_columns, index=X.index)


def get_encoding_pipeline():
    pipeline = Pipeline(steps=[
        ('drop_cols', ColumnDropper(columns_to_drop=['order_id'])),
        ('encode', EncodingTransformer())
    ])
    return pipeline
