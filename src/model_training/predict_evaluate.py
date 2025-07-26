# predict_evaluate.py

import os
import sys
import pandas as pd
import joblib

# path setup
# BASE_DIR  point to the 'RouteWise' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
# ensuring OUTPUT_DIR exists for saving maps
os.makedirs(OUTPUT_DIR, exist_ok=True)

# module imports
sys.path.append(os.path.join(BASE_DIR, 'src', 'feature_engineering'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'model_training'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'map_components'))

# import map_visualisation for direct use
from map_visualisation import visualize_delivery_routes_on_map

# importing neccesary custom classes from saved pipeline
from feature_extraction import TemporalFeatures, HaversineDistance, TimeTakenFeature, DropRawColumns
from outlier import OutlierRemover
from encoding import ColumnStandardizer, ColumnDropper, EncodingTransformer
from scaling import CustomScalerTransformer
from feature_selection import TreeFeatureSelector


# global components (loaded once )
# paths to saved pipeline and model
PREPROCESSING_PIPELINE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')

# global variables to hold loaded components for efficiency
_preprocessing_pipeline = None
_model = None

# functions
def load_inference_components():
    """loads the pre-trained preprocessing pipeline and the best model"""
    global _preprocessing_pipeline, _model # declare intent to modify global variables

    if _preprocessing_pipeline is not None and _model is not None:
        print(" inference components already loaded")
        return True # Components already loaded

    try:
        _preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
        print(f" preprocessing pipeline loaded from: {PREPROCESSING_PIPELINE_PATH}")
    except FileNotFoundError:
        print(f" error: Preprocessing pipeline not found at {PREPROCESSING_PIPELINE_PATH}.")
        print("please run main_pipeline.py first to train and save the pipeline")
        return False
    except Exception as e:
        print(f" error loading preprocessing pipeline: {e}")
        return False
    
    # Try loading the best tuned model (e.g., XGBoost, RandomForest)
    # This logic should match what's saved in main_pipeline.py
    best_model_found = False
    
    # Prioritize XGBoost if it was the best model
    xgb_model_path = os.path.join(MODEL_DIR, 'Best_Tuned_XGBoost_Model.joblib')
    if os.path.exists(xgb_model_path):
        try:
            _model = joblib.load(xgb_model_path)
            print(" best tuned model (XGBoost) loaded")
            best_model_found = True
        except Exception as e:
            print(f" error loading XGBoost model: {e}")
            
    if not best_model_found:
        # fallback to RandomForest if XGBoost wasn't found or failed to load
        rf_model_path = os.path.join(MODEL_DIR, 'Best_Tuned_RandomForest_Model.joblib')
        if os.path.exists(rf_model_path):
            try:
                _model = joblib.load(rf_model_path)
                print(" best tuned model (RandomForest) loaded")
                best_model_found = True
            except Exception as e:
                print(f" error loading RandomForest model: {e}")

    if not best_model_found:
        print(" error: No best tuned model (XGBoost or RandomForest) found in the models directory")
        print("please ensure main_pipeline.py completed successfully and saved a tuned model")
        _preprocessing_pipeline = None # Reset if model not found
        return False

    return True


def get_prediction(user_input_data: dict) -> tuple:
    """
    Takes raw user input, preprocesses it, and predicts delivery time.
    Also returns data suitable for map visualization.
    
    Args:
        user_input_data (dict): A dictionary containing raw input features.
                                This should mirror the structure of the original data.
                                Example: {
                                    'order_id': 'ORD_XYZ',
                                    'order_date': 'YYYY-MM-DD',
                                    'order_time': 'HH:MM:SS',
                                    'pickup_time': 'HH:MM:SS',
                                    'store_latitude': 12.345,
                                    'store_longitude': 78.901,
                                    'drop_latitude': 12.456,
                                    'drop_longitude': 78.012,
                                    'agent_age': 25,
                                    'agent_rating': 4.5,
                                    'traffic': 'medium',
                                    'weather': 'Sunny',
                                    'vehicle': 'Motorcycle',
                                    'area': 'Urban',
                                    'category': 'Food'
                                }

    Returns:
        tuple: (predicted_time_in_minutes, data_for_map_viz_df) if successful,
               (None, None) otherwise.
    """
    if not load_inference_components():
        return None, None

    # define all expected columns for raw input to maintain consistency with training data
    # columns expected by the preprocessing pipeline
    expected_raw_columns = [
        'order_id', # required for map visualization
        'order_date', 'order_time', 'pickup_time',
        'store_latitude', 'store_longitude',
        'drop_latitude', 'drop_longitude',
        'agent_age', 'agent_rating',
        'traffic', 'weather', 'vehicle', 'area', 'category'
    ]

    # create a DataFrame from the single user input
    input_df = pd.DataFrame([user_input_data])
    
    # ensure all expected columns are present, fill missing with NaN if any
    for col in expected_raw_columns:
        if col not in input_df.columns:
            input_df[col] = pd.NA # use pd.NA for nullable data types
    
    # reorder columns to match the expected order of the pipeline's first step
    input_df = input_df[expected_raw_columns]
    input_df.columns = input_df.columns.str.lower() # standardize column names to lowercase

    #  Domain-based Validation (for user input) 
    # as OutlierRemover was removed from the pipeline (due to row dropping),
    if 'agent_age' in input_df.columns and (input_df['agent_age'].iloc[0] < 18 or input_df['agent_age'].iloc[0] > 60):
        print(f" warning: Agent age ({input_df['agent_age'].iloc[0]}) is outside typical operational range (18-60)")
    if 'agent_rating' in input_df.columns and (input_df['agent_rating'].iloc[0] < 1 or input_df['agent_rating'].iloc[0] > 5):
        print(f" warning: Agent rating ({input_df['agent_rating'].iloc[0]}) is outside typical range (1-5)")
    
    # store original coordinates and order_id for map visualization before preprocessing
    data_for_map_viz = input_df[[
        'order_id', 
        'store_latitude', 'store_longitude', 
        'drop_latitude', 'drop_longitude'
    ]].copy()

    try:
        # preprocess the input data using the loaded pipeline
        processed_input = _preprocessing_pipeline.transform(input_df)

        # make prediction
        predicted_time = _model.predict(processed_input)[0]

        # add predicted time to the map visualization data
        data_for_map_viz['predicted_time'] = predicted_time
        
        return predicted_time, data_for_map_viz

    except Exception as e:
        print(f" an error occurred during prediction: {e}")
        import traceback
        traceback.print_exc() # print full traceback for debugging
        return None, None

# example usage (standalone testing)

if __name__ == "__main__":
    print(" Running predict_evaluate.py in standalone mode for testing")

    # attempt to load inference components
    print("\n attempting to load inference components...")
    if not load_inference_components():
        print("exiting as components could not be loaded. Please run main_pipeline.py first.")
        sys.exit(1)

    #  sample user input for testing
    sample_user_input = {
        'order_id': 'PRED_001_JUL_24', # Unique ID for this prediction
        'order_date': '2025-07-24', # Current date
        'order_time': '19:00:00', # 7 PM
        'pickup_time': '19:10:00', # 7:10 PM
        'store_latitude': 12.9716, #  Bangalore city center
        'store_longitude': 77.5946,
        'drop_latitude': 12.9279,  
        'drop_longitude': 77.6271,
        'agent_age': 28,
        'agent_rating': 4.7,
        'traffic': 'High', 
        'weather': 'Cloudy',  
        'vehicle': 'Motorcycle', 
        'area': 'Urban', 
        'category': 'Food'
    }

    print("\nAttempting to predict for sample user input...")
    predicted_delivery_time, map_data_df = get_prediction(sample_user_input)

    if predicted_delivery_time is not None:
        print(f"\n Predicted Delivery Time: {predicted_delivery_time:.2f} minutes")
        print("-- Data for Map Visualization (first 5 rows) --")
        print(map_data_df.head()) 

        # visualize the prediction on map
        print("\n  visualizing predicted route on map...")
        try:
            map_output_path = os.path.join(OUTPUT_DIR, f"predicted_route_{sample_user_input['order_id']}.html")
            visualize_delivery_routes_on_map(
                df=map_data_df,
                pickup_lat_col='store_latitude',
                pickup_lon_col='store_longitude',
                delivery_lat_col='drop_latitude',
                delivery_lon_col='drop_longitude',
                popup_cols=['order_id', 'predicted_time'],
                map_title=f"Predicted Route for {sample_user_input['order_id']} (Time: {predicted_delivery_time:.2f} min)",
                output_html_path=map_output_path
            )
            print(f" map generated successfully at: {map_output_path}")
        except Exception as e:
            print(f" error during map visualization: {e}")
            print("please ensure your map_visualisation.py is correctly set up and paths are accessible")
    else:
        print("\n Prediction failed. Please check the error messages above.")

    print("\n--- predict_evaluate.py standalone execution complete ---")
