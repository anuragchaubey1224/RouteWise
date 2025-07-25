# main_pipeline.py

import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# path setup for the main pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'delivery_data_cleaned.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#  module imports (for all pipeline components)
sys.path.append(os.path.join(BASE_DIR, 'src', 'feature_engineering'))

from feature_extraction import TemporalFeatures, HaversineDistance, TimeTakenFeature, DropRawColumns
from outlier import OutlierRemover
from encoding import ColumnStandardizer, ColumnDropper, EncodingTransformer
from scaling import CustomScalerTransformer
from feature_selection import TreeFeatureSelector

sys.path.append(os.path.join(BASE_DIR, 'src', 'model_training'))
from train_model import train_multiple_models, tune_and_save_best_model


#  main function to run the full pipeline
def run_full_pipeline():
    """
    execute the full pipeline:
    1. data loading
    2. feature extraction
    3. outlier removal
    4. encoding categorical features
    5. scaling numerical features
    6. selects important features
    7. train and tune models
    8. saves the full inference pipeline
    """
    # 1. data loading
    print(" loading cleaned delivery data...")
    if not os.path.exists(DATA_PATH):
        print(f" Error: Data file not found at {DATA_PATH}")
        print("check the path and ensure the data is available")
        return
        
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    print(f"initial data loaded with shape: {df.shape}")

    # original coordinates and order_id  stored before transformation for map visualization later
    initial_coordinates_df = df[['order_id', 'store_latitude', 'store_longitude', 'drop_latitude', 'drop_longitude']].copy()


    # separate features (X) and target variable (y)
    TARGET_COLUMN = 'delivery_time'
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN]
        X = df.drop(columns=[TARGET_COLUMN])
    else:
        print(f" Error: target column '{TARGET_COLUMN}' not found in data ")
        return

    #  initializing individual pipeline components for full inference pipeline
    # OutlierRemover is excluded from this sequential pipeline for inference

    #  feature extraction steps (Temporal, Haversine, TimeTaken, DropRawColumns)
    temporal_features_step = TemporalFeatures()
    haversine_distance_step = HaversineDistance()
    time_taken_feature_step = TimeTakenFeature()
    drop_raw_columns_step = DropRawColumns() # will be used in the final saved pipeline

    # encoding steps
    column_standardizer_step = ColumnStandardizer()
    column_dropper_step = ColumnDropper(columns_to_drop=['order_id'])
    encoding_transformer_step = EncodingTransformer()
    
    # scaling step
    custom_scaler_transformer_step = CustomScalerTransformer()

    # feature selection step
    tree_feature_selector_step = TreeFeatureSelector(n_features_to_select=10)


    # 2. feature extraction 
    print("\n running feature extraction pipeline...")
    
    # applying individual feature extraction steps
    X_transformed = temporal_features_step.fit_transform(X.copy())
    X_transformed = haversine_distance_step.fit_transform(X_transformed)
    X_transformed = time_taken_feature_step.fit_transform(X_transformed)
    
    # drop raw columns after feature extraction is done for training data
    X_transformed = drop_raw_columns_step.fit_transform(X_transformed) # apply DropRawColumns here

    # save intermediate feature extraction results
    feature_path = os.path.join(OUTPUT_DIR, 'feature_extraction.csv')
    pd.concat([X_transformed, y.reset_index(drop=True).rename(TARGET_COLUMN)], axis=1).to_csv(feature_path, index=False)
    print(f" feature extraction completed. Shape: {X_transformed.shape}. Saved to:\n   {feature_path}")
    
    #  3. outlier removal 
    print("\n removing outliers...")
    outlier_remover = OutlierRemover()
    df_for_outliers = pd.concat([X_transformed, y.rename('temp_target_for_outliers')], axis=1)
    df_outlier_removed_full = outlier_remover.fit_transform(df_for_outliers)

    y_outlier_removed = df_outlier_removed_full['temp_target_for_outliers'].rename(TARGET_COLUMN).reset_index(drop=True)
    X_outlier_removed = df_outlier_removed_full.drop(columns=['temp_target_for_outliers']).reset_index(drop=True)
    
    outlier_path = os.path.join(OUTPUT_DIR, 'outlier.csv')
    pd.concat([X_outlier_removed, y_outlier_removed], axis=1).to_csv(outlier_path, index=False)
    print(f" outliers removed. Shape: {X_outlier_removed.shape}. Saved to:\n   {outlier_path}")
    
    # 4. encoding 
    print("\n running encoding pipeline...")
    # apply individual encoding steps
    X_encoded_standardized = column_standardizer_step.fit_transform(X_outlier_removed.copy())
    X_encoded_dropped = column_dropper_step.fit_transform(X_encoded_standardized)
    X_encoded = encoding_transformer_step.fit_transform(X_encoded_dropped)

    encoded_path = os.path.join(OUTPUT_DIR, 'encoded.csv')
    pd.concat([X_encoded, y_outlier_removed], axis=1).to_csv(encoded_path, index=False)
    print(f" encoding completed. Shape: {X_encoded.shape}. Saved to:\n   {encoded_path}")
    
    # 5. scaling 
    print("\n running scaling pipeline...")
    X_scaled = custom_scaler_transformer_step.fit_transform(X_encoded)
    scaled_path = os.path.join(OUTPUT_DIR, "scaled.csv")
    pd.concat([X_scaled, y_outlier_removed], axis=1).to_csv(scaled_path, index=False)
    print(f" scaling completed. Shape: {X_scaled.shape}. Saved to:\n   {scaled_path}")

    # ensure all columns are numeric before feature selection
    print("\n checking for non-numeric columns before feature selection...")
    non_numeric_cols = X_scaled.select_dtypes(include=['object', 'datetime64[ns]', 'timedelta64[ns]']).columns.tolist()
    if non_numeric_cols:
        print(f" warning: found non-numeric columns in X_scaled: {non_numeric_cols}")
        print("dropping non-numeric columns")
        X_scaled = X_scaled.drop(columns=non_numeric_cols)
        print(f"Non-numeric columns dropped  new shape of X_scaled: {X_scaled.shape}")
    else:
        print(" all columns in X_scaled are numeric , proceeding to feature selection")
    
    # 6. Feature Selection 
    print("\n running feature selection...")
    X_selected = tree_feature_selector_step.fit_transform(X_scaled, y_outlier_removed)
    
    selected_path = os.path.join(OUTPUT_DIR, "selected_features.csv")
    pd.concat([X_selected, y_outlier_removed], axis=1).to_csv(selected_path, index=False)
    print(f" feature selection completed. Shape: {X_selected.shape}. Saved to:\n   {selected_path}")

    #  7. model training and tuning 
    print("\n starting model training and tuning...")
    df_for_training = pd.read_csv(selected_path)
    
    initial_shape = df_for_training.shape
    df_for_training.dropna(inplace=True)
    if df_for_training.shape != initial_shape:
        print(f" dropped {initial_shape[0] - df_for_training.shape[0]} rows containing NaN values for model training")
        print(f"new data shape after NaN removal: {df_for_training.shape}")
    
    training_results = train_multiple_models(
        df=df_for_training,
        target_column="delivery_time",
        do_cross_validation=True,
        cv_folds=5
    )

    best_rmse = float('inf')
    best_model_name = None
    for name, metrics in training_results["performance_metrics"].items():
        current_rmse = metrics.get('CV_RMSE_Mean', metrics['RMSE'])
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_model_name = name
    
    print(f"\n best initial model (based on RMSE): {best_model_name} with RMSE: {best_rmse:.2f}")

    # Tune the best model and get the final best estimator
    tune_and_save_best_model(
        X_train=training_results["X_train"],
        y_train=training_results["y_train"],
        X_test=training_results["X_test"],
        y_test=training_results["y_test"],
        best_model_name=best_model_name
    )
    print("\n model training and tuning completed.")

    # load the best tuned model to include in the full inference pipeline
    best_model_path = os.path.join(MODEL_DIR, f"Best_Tuned_{best_model_name}_Model.joblib")
    if os.path.exists(best_model_path):
        final_best_model = joblib.load(best_model_path)
        print(f" loaded best tuned model: {best_model_name}")
    else:
        print(f" error: Best tuned model not found at {best_model_path}  cannot create full inference pipeline")
        return

    #  8. create and Save full inference pipeline ---
    print("\n  creating and saving the full inference pipeline...")

    # define the preprocessing pipeline using the *fitted instances*
    # this pipeline will be loaded by predict_evaluate.py for inference
    preprocessing_pipeline = Pipeline(steps=[
        ('temporal', temporal_features_step),
        ('haversine', haversine_distance_step),
        ('timediff', time_taken_feature_step),
        ('standardize', column_standardizer_step),
        ('drop_id', column_dropper_step), 
        ('encode', encoding_transformer_step), 
        ('scale', custom_scaler_transformer_step),
        ('select_features', tree_feature_selector_step),
        # addinf a final DropRawColumns to ensure raw columns are dropped for inference
        ('drop_raw_for_inference', DropRawColumns())
    ])
    
    # save the preprocessing pipeline
    preprocessing_pipeline_path = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')
    try:
        joblib.dump(preprocessing_pipeline, preprocessing_pipeline_path)
        print(f" preprocessing pipeline saved to: {preprocessing_pipeline_path}")
    except Exception as e:
        print(f" error saving preprocessing pipeline: {e}")

    print("\n full pipeline executed successfully!")


# entry point
if __name__ == "__main__":
    run_full_pipeline()
