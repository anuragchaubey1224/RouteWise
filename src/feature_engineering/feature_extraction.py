import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Temporal Feature Extraction 
class TemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        try:
            order_time = pd.to_datetime(X['order_time'], errors='coerce').dt.strftime('%H:%M:%S')
            pickup_time = pd.to_datetime(X['pickup_time'], errors='coerce').dt.strftime('%H:%M:%S')
            X['order_datetime'] = pd.to_datetime(X['order_date'] + ' ' + order_time, errors='coerce')
            X['pickup_datetime'] = pd.to_datetime(X['order_date'] + ' ' + pickup_time, errors='coerce')
        except Exception as e:
            print("Datetime parsing error:", e)
            raise

        X['order_hour'] = X['order_datetime'].dt.hour
        X['order_minute'] = X['order_datetime'].dt.minute
        X['order_dayofweek'] = X['order_datetime'].dt.dayofweek
        X['is_weekend'] = X['order_dayofweek'].isin([5, 6]).astype(int)
        X['is_peakhour'] = X['order_hour'].isin([8, 9, 18, 19]).astype(int)

        X['pickup_hour'] = X['pickup_datetime'].dt.hour
        X['pickup_minute'] = X['pickup_datetime'].dt.minute

        return X

#  Haversine Distance 
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Radius of Earth in km

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

#  Time Difference 
class TimeTakenFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['order_to_pickup_min'] = (X['pickup_datetime'] - X['order_datetime']).dt.total_seconds() / 60.0
        return X

#  Delivery Speed
class DeliverySpeedFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['delivery_speed_km_min'] = X['haversine_distance_km'] / (X['delivery_time'] + 1e-6)
        return X

#  Drop Raw Columns 
class DropRawColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols_to_drop = [
            'order_date', 'order_time', 'pickup_time',
            'store_latitude', 'store_longitude',
            'drop_latitude', 'drop_longitude',
            'order_datetime', 'pickup_datetime'
        ]
        existing_cols = [col for col in cols_to_drop if col in X.columns]
        return X.drop(columns=existing_cols)

#  Feature Pipeline 
feature_pipeline = Pipeline([
    ('temporal', TemporalFeatures()),
    ('haversine', HaversineDistance()),
    ('timediff', TimeTakenFeature()),
    ('speed', DeliverySpeedFeature()),
    ('dropcols', DropRawColumns())
])

#  Run Script 
if __name__ == "__main__":
    input_path = "/Users/anuragchaubey/RouteWise/data/processed/delivery_data_cleaned.csv"
    output_path = "/Users/anuragchaubey/RouteWise/data/processed/delivery_data_features.csv"

    df = pd.read_csv(input_path)

    # Clean column names (lowercase)
    df.columns = df.columns.str.lower()

    # Apply feature extraction pipeline
    try:
        df_features = feature_pipeline.fit_transform(df)
        df_features.to_csv(output_path, index=False)
        print(f" Feature extraction complete. File saved to:\n{output_path}")
    except Exception as e:
        print(" Error during feature extraction:", e)
