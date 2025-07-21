# encoding.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class CyclicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col=None, max_val=None):
        if not col or max_val is None:
            raise ValueError("CyclicEncoder requires 'col' and 'max_val'")
        self.col = col
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.col in X.columns:
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

        self.ordinal_encoder = OrdinalEncoder(
            categories=[['low', 'medium', 'high', 'jam']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.onehot_encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )
        self.cyclic_transformer = CyclicEncoder(col=self.cyclic_col, max_val=6)
        self.column_transformer = None
        self.feature_names_out = None

    def fit(self, X, y=None):
        X_temp = self.cyclic_transformer.transform(X.copy())

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('ord', self.ordinal_encoder, [col for col in self.ordinal_cols if col in X_temp.columns]),
                ('ohe', self.onehot_encoder, [col for col in self.onehot_cols if col in X_temp.columns])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        self.column_transformer.fit(X_temp)
        self.feature_names_out = self.column_transformer.get_feature_names_out()
        return self

    def transform(self, X):
        X_transformed = self.cyclic_transformer.transform(X.copy())
        transformed_array = self.column_transformer.transform(X_transformed)
        return pd.DataFrame(transformed_array, columns=self.feature_names_out, index=X.index)


def get_encoding_pipeline():
    return Pipeline(steps=[
        ('drop_cols', ColumnDropper(columns_to_drop=['order_id'])),
        ('encode', EncodingTransformer())
    ])
