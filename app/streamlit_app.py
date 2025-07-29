# RouteWise/apps/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from streamlit_lottie import st_lottie
import requests 

# to handle FutureWarnings from seaborn and sklearn
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

# base directory and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'delivery_data_cleaned.csv') # Data is now on GitHub
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # Models are now on GitHub
EDA_NOTEBOOK_PATH = os.path.join(BASE_DIR, 'eda_analysis.ipynb') # Assuming it's in the root

# ensure output directory exists for map HTML files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# add 'src' directory to sys.path to import custom modules
sys.path.append(os.path.join(BASE_DIR, 'src'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'model_training'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'map_components'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'feature_engineering')) # required for custom transformers

# import the get_prediction function from predict_evaluate.py
try:
    from predict_evaluate import get_prediction
    from map_visualisation import visualize_delivery_routes_on_map
    # import custom transformers to ensure joblib can load the pipeline
    from feature_extraction import TemporalFeatures, HaversineDistance, TimeTakenFeature, DropRawColumns
    from outlier import OutlierRemover
    from encoding import ColumnStandardizer, ColumnDropper, EncodingTransformer
    from scaling import CustomScalerTransformer
    from feature_selection import TreeFeatureSelector
except ImportError as e:
    st.error(f"Error loading core modules: {e}. Please ensure all Python files are in their correct 'src' subdirectories and your environment is set up correctly.")
    st.stop() # stop the app if core modules can't be imported

# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="wide", # Use wide layout for better display
    initial_sidebar_state="expanded"
)

#  CACHING DATA AND COMPONENTS 
@st.cache_data
def load_data():
    """loads the cleaned delivery data for EDA"""
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.lower() # Standardize column names
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at `{DATA_PATH}`. Please ensure 'delivery_data_cleaned.csv' is in the `data/processed` directory on GitHub.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_lottieurl(url: str):
    """Loads Lottie animation data from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations
lottie_delivery_truck = load_lottieurl("https://lottie.host/17808269-026d-473d-986c-7e9b086e1088/V9d04G6r5X.json")
lottie_data_analysis = load_lottieurl("https://lottie.host/0a9c6c8f-3957-4186-9a2d-11487224c65f/xO6L6zC9r7.json")
lottie_prediction = load_lottieurl("https://lottie.host/43a509f6-0248-406c-8207-62f4007358a9/T58X79m43x.json")
lottie_model = load_lottieurl("https://lottie.host/801b6375-7286-4f51-b84e-2895f32a2656/W761vI8r5E.json")


#  PAGE FUNCTIONS 

def home_page():
    st.title("📦 Logistics Delivery Time Prediction")
    st.markdown("""
    This Streamlit web application showcases a machine learning model for predicting delivery time using a comprehensive logistics dataset.
    This project focuses on the importance of accurately estimating delivery times based on various operational and environmental factors.
    """)

    st.header("Project Objective 🎯")
    st.markdown("""
    In the e-commerce and logistics industry, accurate delivery time estimation is crucial for customer satisfaction and operational efficiency.
    This app addresses this challenge by analyzing historical delivery records and building a machine learning model
    that can predict delivery times for new orders.
    """)

    # Lottie animation for visual appeal
    if lottie_delivery_truck:
        st_lottie(lottie_delivery_truck, height=200, key="delivery_truck_animation")
    else:
        st.image("https://placehold.co/600x300/ADD8E6/000000?text=Delivery+Time+Prediction", caption="Delivery Time Prediction")

    st.header("Key Features ✨")
    st.markdown("""
    * **Exploratory Data Analysis (EDA):** Interactive visualizations for a deep understanding of the dataset.
    * **Delivery Time Prediction:** Provide inputs to get an estimated delivery time for new orders.
    * **Route Visualization:** View the predicted delivery route on a map, showing pickup and delivery locations.
    * **Model Methodology:** Learn about the machine learning pipeline and model performance used.
    """)
    
    st.info("Use the sidebar on the left to navigate. 👈")

def eda_page():
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section presents key insights and analyses performed on the logistics delivery dataset.
    """)

    df = load_data()
    if df.empty:
        st.warning("Could not load data. Please ensure 'delivery_data_cleaned.csv' is in the correct location.")
        return

    # Lottie animation for EDA
    if lottie_data_analysis:
        st_lottie(lottie_data_analysis, height=150, key="eda_animation")

    st.header("Dataset Overview 📋")
    st.subheader("First 5 Rows of Data")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    st.subheader("Column Information and Data Types")
    # using a string buffer to capture df.info() output
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Descriptive Statistics for Numerical Features")
    st.dataframe(df.describe())

    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if not missing_data.empty:
        st.write("Columns with Missing Values:")
        st.dataframe(missing_data)
        st.info("Missing values were handled during the data cleaning phase (e.g., imputation or removal).")
    else:
        st.success("No missing values found in the cleaned dataset. ✨")

    st.header("Key Visualizations 📈")

    # use st.expander for better organization
    with st.expander("Distribution of Delivery Time ⏱️"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['delivery_time'], kde=True, ax=ax, color='skyblue')
        ax.set_title('Distribution of Delivery Time')
        ax.set_xlabel('Delivery Time (minutes)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        st.markdown("**Insight:** Delivery time is generally well-distributed, with most deliveries falling between 90 and 160 minutes.")
        plt.close(fig) # close plot to free memory

    with st.expander("Distribution of Agent Age 🧑‍✈️"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['agent_age'], kde=True, ax=ax, color='lightcoral')
        ax.set_title('Distribution of Agent Age')
        ax.set_xlabel('Agent Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        st.markdown("**Insight:** Most delivery agents are between 20 and 35 years old.")
        plt.close(fig)

    with st.expander("Distribution of Agent Rating ⭐"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['agent_rating'], kde=True, ax=ax, color='lightgreen')
        ax.set_title('Distribution of Agent Rating')
        ax.set_xlabel('Agent Rating')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        st.markdown("**Insight:** The majority of agents have ratings between 4.0 and 5.0, indicating high customer satisfaction.")
        plt.close(fig)

    with st.expander("Delivery Time by Traffic Condition 🚦"):
        fig, ax = plt.subplots(figsize=(10, 6))
        # fixed FutureWarning by assigning hue
        sns.boxplot(x='traffic', y='delivery_time', data=df, ax=ax, palette='viridis', hue='traffic', legend=False)
        ax.set_title('Delivery Time by Traffic Condition')
        ax.set_xlabel('Traffic')
        ax.set_ylabel('Delivery Time (minutes)')
        st.pyplot(fig)
        st.markdown("**Insight:** Delivery time significantly increases during 'Jam' traffic conditions, while 'Low' traffic results in the shortest times.")
        plt.close(fig)

    with st.expander("Delivery Time by Weather Condition ☁️"):
        fig, ax = plt.subplots(figsize=(10, 6))
        # fixed FutureWarning by assigning hue
        sns.boxplot(x='weather', y='delivery_time', data=df, ax=ax, palette='plasma', hue='weather', legend=False)
        ax.set_title('Delivery Time by Weather Condition')
        ax.set_xlabel('Weather')
        ax.set_ylabel('Delivery Time (minutes)')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("**Insight:** Adverse weather conditions such as 'Stormy' and 'Sandstorms' lead to increased delivery times.")
        plt.close(fig)

    with st.expander("Delivery Time by Vehicle Type 🏍️"):
        fig, ax = plt.subplots(figsize=(10, 6))
        # fixed FutureWarning by assigning hue
        sns.boxplot(x='vehicle', y='delivery_time', data=df, ax=ax, palette='cividis', hue='vehicle', legend=False)
        ax.set_title('Delivery Time by Vehicle Type')
        ax.set_xlabel('Vehicle')
        ax.set_ylabel('Delivery Time (minutes)')
        st.pyplot(fig)
        st.markdown("**Insight:** 'Motorcycle' and 'Scooter' generally show faster delivery times compared to 'Van'.")
        plt.close(fig)

    with st.expander("Feature Correlation Heatmap 🔥"):
        numeric_df = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap of Numerical Features')
        st.pyplot(fig)
        st.markdown("**Insight:** There is a strong positive correlation between `haversine_distance_km` and `delivery_time`, which is expected.")
        plt.close(fig)


def prediction_page():
    st.title("⚡ Predict Delivery Time")
    st.markdown("Please enter the delivery order details for the model to estimate the delivery time.")

    # lottie animation for Prediction
    if lottie_prediction:
        st_lottie(lottie_prediction, height=150, key="prediction_animation")

    # input form layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Order Details 📝")
        order_id = st.text_input("Order ID", "ORD_" + str(np.random.randint(100000, 999999)))
        order_date = st.date_input("Order Date", pd.to_datetime('today'))
        order_time = st.time_input("Order Time", pd.to_datetime('19:00:00').time())
        pickup_time = st.time_input("Pickup Time", pd.to_datetime('19:10:00').time())

        st.subheader("Location Details 📍")
        store_latitude = st.number_input("Store Latitude", value=12.9716, format="%.6f")
        store_longitude = st.number_input("Store Longitude", value=77.5946, format="%.6f")
        drop_latitude = st.number_input("Drop Latitude", value=12.9279, format="%.6f")
        drop_longitude = st.number_input("Drop Longitude", value=77.6271, format="%.6f")

    with col2:
        st.subheader("Agent and Environmental Factors 🧑‍💻")
        agent_age = st.number_input("Agent Age", min_value=18, max_value=60, value=28)
        agent_rating = st.number_input("Agent Rating", min_value=1.0, max_value=5.0, value=4.7, step=0.1)

        # categorical features - ensure these match your training data categories
        traffic_options = ['Low', 'Medium', 'High', 'Jam']
        traffic = st.selectbox("Traffic Condition", traffic_options, index=traffic_options.index('High'))

        weather_options = ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Sandstorms', 'Fog']
        weather = st.selectbox("Weather Condition", weather_options, index=weather_options.index('Cloudy'))

        vehicle_options = ['Motorcycle', 'Scooter', 'Electric Scooter', 'Bicycle', 'Van']
        vehicle = st.selectbox("Vehicle Type", vehicle_options, index=vehicle_options.index('Motorcycle'))

        area_options = ['Urban', 'Metropolitian', 'Semi-Urban']
        area = st.selectbox("Area", area_options, index=area_options.index('Urban'))

        category_options = ['Food', 'Drinks', 'Snacks', 'Presents', 'Desserts', 'Buffet', 'Meal', 'Other', 'Clothing', 'Electronics', 'Apparel']
        category = st.selectbox("Category", category_options, index=category_options.index('Food'))

    # prediction Button
    st.markdown("---")
    if st.button("Estimate Delivery Time 🚀"):
        with st.spinner("Calculating prediction... ⏳"):
            user_input = {
                'order_id': order_id,
                'order_date': order_date.strftime('%Y-%m-%d'),
                'order_time': order_time.strftime('%H:%M:%S'),
                'pickup_time': pickup_time.strftime('%H:%M:%S'),
                'store_latitude': store_latitude,
                'store_longitude': store_longitude,
                'drop_latitude': drop_latitude,
                'drop_longitude': drop_longitude,
                'agent_age': agent_age,
                'agent_rating': agent_rating,
                'traffic': traffic,
                'weather': weather,
                'vehicle': vehicle,
                'area': area,
                'category': category
            }

            predicted_time, map_data_df = get_prediction(user_input)

            if predicted_time is not None:
                st.success(f"**Estimated Delivery Time:** {predicted_time:.2f} minutes 🎉")
                
                # display Map
                st.subheader("Delivery Route on Map 🗺️")
                map_output_path = os.path.join(OUTPUT_DIR, f"predicted_route_{order_id}.html")
                
                visualize_delivery_routes_on_map(
                    df=map_data_df,
                    pickup_lat_col='store_latitude',
                    pickup_lon_col='store_longitude',
                    delivery_lat_col='drop_latitude',
                    delivery_lon_col='drop_longitude',
                    popup_cols=['order_id', 'predicted_time'],
                    map_title=f"Predicted Route for Order {order_id} (Time: {predicted_time:.2f} min)",
                    output_html_path=map_output_path
                )

                # embed the generated HTML map
                if os.path.exists(map_output_path):
                    with open(map_output_path, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    st.components.v1.html(map_html, height=500, scrolling=True)
                else:
                    st.error("Map HTML file could not be generated. 😔")
            else:
                st.error("An error occurred during prediction. Please check inputs and ensure the model and pipeline are loaded correctly. ❌")

def model_methodology_page():
    st.title("🧠 Model and Methodology")
    st.markdown("""
    In this section, we will delve into the machine learning pipeline and model used for delivery time prediction.
    """)

    # lottie animation for Model & Methodology
    if lottie_model:
        st_lottie(lottie_model, height=150, key="model_animation")

    st.header("1. Data Processing Pipeline ⚙️")
    st.markdown("""
    The data undergoes several stages to ensure accurate delivery time prediction:

    * **Feature Extraction:** Creating new, more informative features from raw data (e.g., temporal features, Haversine distance).
    * **Outlier Removal:** Identifying and removing outliers (anomalous values) to improve model performance.
    * **Encoding:** Converting categorical features into numerical representations (e.g., one-hot encoding).
    * **Scaling:** Bringing numerical features to a common scale so that models can perform better.
    * **Feature Selection:** Selecting the most relevant features based on their predictive power, reducing model complexity and improving performance.
    """)
    st.image("https://placehold.co/600x200/ADD8E6/000000?text=ML+Pipeline+Flowchart", caption="Machine Learning Pipeline Flowchart (Example)")
    st.markdown("---")

    st.header("2. Model Training and Tuning 🚀")
    st.markdown("""
    We evaluated several regression models for delivery time prediction.
    **XGBoost Regressor** was chosen as the best model due to its excellent performance and interpretability.

    **XGBoost (eXtreme Gradient Boosting):**
    It is a powerful, scalable, and accurate machine learning technique used for both classification and regression tasks. It is an implementation of boosted trees and is known for its speed and performance.

    **Hyperparameter Tuning:**
    Hyperparameter tuning was performed using Grid Search to optimize the model's performance.
    """)

    st.subheader("Performance of the Best Tuned Model (XGBoost) 📊")
    st.markdown("""
    Metrics obtained from your last `main_pipeline.py` run:
    """)
    st.write("MAE: **17.26**")
    st.write("RMSE: **22.30**")
    st.write("R²: **0.8126**")
    st.info("These metrics indicate that the model is quite accurate in predicting delivery time on the test set. ✅")

    st.subheader("Feature Importance 💡")
    st.markdown("""
    Top 10 most important features for the model:
    """)
    
    # attempt to load the model to get feature importances
    try:
        model_path = os.path.join(MODEL_DIR, 'Best_Tuned_XGBoost_Model.joblib')
        if os.path.exists(model_path):
            best_model = joblib.load(model_path)
            if hasattr(best_model, 'feature_importances_'):
                if hasattr(best_model, 'feature_names_in_') and best_model.feature_names_in_ is not None:
                    feature_names = best_model.feature_names_in_
                else:
                    # fallback if feature_names_in_ is not directly available
                    feature_names = [f"Feature_{i+1}" for i in range(len(best_model.feature_importances_))] 
                    
                importances = best_model.feature_importances_
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(10, 6))
                # Fixed FutureWarning by assigning hue
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='viridis', hue='Feature', legend=False)
                ax.set_title('Feature Importance')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Feature importances are not available or the model does not have a 'feature_importances_' attribute. ℹ️")
        else:
            st.warning("Best tuned model not found. Feature importance cannot be displayed. ⚠️")
    except Exception as e:
        st.error(f"Error loading or displaying feature importances: {e} ❌")
        st.info("This typically happens if the model cannot be loaded or does not have a 'feature_importances_' attribute.")


def about_page():
    st.title("ℹ️ About the Project")
    st.header("Developer 👨‍💻")
    st.markdown("""
    **Anurag Chaubey**
    * [LinkedIn](https://www.linkedin.com/in/anurag-chaubey-63202a297?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
    * [GitHub](https://github.com/anuragchaubey1224)
    """)

    st.header("Technologies Used 🛠️")
    st.markdown("""
    * **Programming Language:** Python
    * **Data Manipulation:** Pandas, NumPy
    * **Machine Learning:** Scikit-learn, XGBoost
    * **Visualization:** Matplotlib, Seaborn, Folium
    * **Web App Framework:** Streamlit, Streamlit-Lottie
    """)

    st.header("GitHub Repository 🔗")
    st.markdown("[RouteWise GitHub Repository](https://github.com/anuragchaubey101/RouteWise)")

    st.header("Future Enhancements 🚀")
    st.markdown("""
    * **Real-time Data Integration:** Integrate real-time delivery data.
    * **More Advanced Models:** Explore deep learning models or ensemble methods.
    * **UI/UX Improvements:** Further enhance the user interface and experience of the app.
    * **Deployment:** Deploy the app to a cloud platform to make it publicly accessible.
    """)

# ---------------- MAIN APP LAYOUT ----------------

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to Page", ["Home", "EDA", "Prediction", "Model & Methodology", "About"])

    if page == "Home":
        home_page()
    elif page == "EDA":
        eda_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Model & Methodology":
        model_methodology_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
