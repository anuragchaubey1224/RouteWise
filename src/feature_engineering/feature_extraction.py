import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# temporal feature extraction
class TemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # parse datetime columns
        order_time = pd.to_datetime(X['order_time']).dt.strftime('%H:%M:%S')
        pickup_time = pd.to_datetime(X['pickup_time']).dt.strftime('%H:%M:%S')

        X['order_datetime'] = pd.to_datetime(X['order_date'] + ' ' + order_time, errors='coerce')
        X['pickup_datetime'] = pd.to_datetime(X['order_date'] + ' ' + pickup_time, errors='coerce')

        # extract time-based features
        X['order_hour'] = X['order_datetime'].dt.hour
        X['order_minute'] = X['order_datetime'].dt.minute
        X['order_dayofweek'] = X['order_datetime'].dt.dayofweek
        X['is_weekend'] = X['order_dayofweek'].isin([5, 6]).astype(int)
        X['is_peakhour'] = X['order_hour'].isin([8, 9, 18, 19]).astype(int)
        X['pickup_hour'] = X['pickup_datetime'].dt.hour
        X['pickup_minute'] = X['pickup_datetime'].dt.minute

        return X

# haversine distance calculation
def haversine_np(lon1, lat1, lon2, lat2):
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth's radius 

class HaversineDistance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['haversine_distance_km'] = haversine_np(
            X['store_longitude'], X['store_latitude'],
            X['drop_longitude'], X['drop_latitude']
        )
        return X

# time taken feature
class TimeTakenFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['order_to_pickup_min'] = (X['pickup_datetime'] - X['order_datetime']).dt.total_seconds() / 60.0
        return X

# drop raw columns
class DropRawColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop_cols = [
            'order_date', 'order_time', 'pickup_time',
            'store_latitude', 'store_longitude',
            'drop_latitude', 'drop_longitude',
            'order_datetime', 'pickup_datetime'
        ]
        return X.drop(columns=[col for col in drop_cols if col in X.columns])

# feature extraction pipeline
feature_pipeline = Pipeline([
    ('temporal', TemporalFeatures()),
    ('haversine', HaversineDistance()),
    ('timediff', TimeTakenFeature()),
    ('dropcols', DropRawColumns())
])

# main pipeline entry point
if __name__ == "__main__":
    print("feature extraction module : ")

