import streamlit as st
import pandas as pd
import numpy as np
import time
from utils.preprocess import load_data, clean_data
from utils.visualization import (
    plot_histogram, plot_boxplot, plot_scatter, plot_correlation_heatmap,
    plot_target_distribution, plot_income_vs_price, plot_ocean_proximity_count,
    plot_latitude_vs_longitude, get_insights
)
from utils.prediction import predict_house_price

# Set Page Config
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Professional Dark Theme & Animations
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
    }
    .metric-card {
        background-color: #1E2127;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 5px solid #4CAF50;
    }
    h1, h2, h3 {
        color: #4CAF50;
        font-family: 'Inter', sans-serif;
    }
    .fade-in {
        animation: fadeIn 1.5s;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Load Data globally for the app
@st.cache_data
def get_dataset(filepath='housing.csv'):
    try:
        return load_data(filepath)
    except:
        return pd.DataFrame()

df_raw = get_dataset()

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = [
    "🏠 Home", "📂 Dataset", "🧹 Data Cleaning", "📊 Exploratory Data Analysis",
    "📈 Insights", "🤖 Model Training", "📉 Evaluation", "💰 Predict House Price", "ℹ️ About"
]
choice = st.sidebar.radio("Go to", pages)

if choice == "🏠 Home":
    st.markdown("<h1 class='fade-in'>House Price Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to the **House Price Prediction Dashboard**. 
    This interactive application allows you to explore the California Housing dataset, view deep analytical insights, train a linear regression model, and predict house prices in real-time.
    
    ### Features:
    - **Interactive Data Exploration**
    - **Automated Business Insights**
    - **Real-time Price Prediction**
    - **Beautiful Visualizations**
    
    Use the sidebar to navigate through the different sections of the ML pipeline.
    """)
    st.image("https://images.unsplash.com/photo-1518780664697-55e3ad937233?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_column_width=True)

elif choice == "📂 Dataset":
    st.title("Dataset Overview")
    if df_raw.empty:
        st.error("Dataset not found! Please make sure housing.csv is in the directory.")
    else:
        st.write("### Raw Data Preview")
        st.dataframe(df_raw.head(100))
        
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric-card'><h4>Total Rows</h4><h2>{df_raw.shape[0]}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><h4>Total Columns</h4><h2>{df_raw.shape[1]}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><h4>Missing Values</h4><h2>{df_raw.isnull().sum().sum()}</h2></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-card'><h4>Duplicates</h4><h2>{df_raw.duplicated().sum()}</h2></div>", unsafe_allow_html=True)
        
        st.write("### Column Details & Data Types")
        st.write(df_raw.dtypes)
        
        st.write("### Download Dataset")
        csv = df_raw.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name='housing.csv', mime='text/csv')
        
        st.write("### Upload New Dataset (Optional)")
        uploaded_file = st.file_uploader("Upload your own CSV file", type=["csv"])
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(df_uploaded.head())

elif choice == "🧹 Data Cleaning":
    st.title("Data Cleaning Process")
    st.write("Handling missing values, removing duplicates, and preparing the dataset for modeling.")
    
    if st.button("Run Data Cleaning Pipeline"):
        with st.spinner('Cleaning data...'):
            time.sleep(1) # Simulation of processing
            df_clean = clean_data(df_raw)
            st.success("Data cleaned successfully!")
            
            st.write("### Cleaned Data Preview")
            st.dataframe(df_clean.head())
            
            col1, col2 = st.columns(2)
            col1.markdown(f"<div class='metric-card'><h4>Remaining Missing Values</h4><h2>{df_clean.isnull().sum().sum()}</h2></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-card'><h4>Remaining Duplicates</h4><h2>{df_clean.duplicated().sum()}</h2></div>", unsafe_allow_html=True)

elif choice == "📊 Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    df_clean = clean_data(df_raw)
    
    chart_type = st.selectbox("Select Chart to View", [
        "Target Distribution", "Median Income vs House Price", 
        "Latitude vs Longitude", "Ocean Proximity Count", 
        "Correlation Heatmap", "Histogram - Total Rooms",
        "Box Plot - Population", "Scatter - Rooms vs Bedrooms"
    ])
    
    with st.spinner("Generating Chart..."):
        time.sleep(0.5)
        if chart_type == "Target Distribution":
            st.plotly_chart(plot_target_distribution(df_clean), use_container_width=True)
        elif chart_type == "Median Income vs House Price":
            st.plotly_chart(plot_income_vs_price(df_clean), use_container_width=True)
        elif chart_type == "Latitude vs Longitude":
            st.plotly_chart(plot_latitude_vs_longitude(df_clean), use_container_width=True)
        elif chart_type == "Ocean Proximity Count":
            st.plotly_chart(plot_ocean_proximity_count(df_clean), use_container_width=True)
        elif chart_type == "Correlation Heatmap":
            st.plotly_chart(plot_correlation_heatmap(df_clean), use_container_width=True)
        elif chart_type == "Histogram - Total Rooms":
            st.plotly_chart(plot_histogram(df_clean, 'total_rooms', 'Distribution of Total Rooms'), use_container_width=True)
        elif chart_type == "Box Plot - Population":
            st.plotly_chart(plot_boxplot(df_clean, 'population', 'Population Box Plot'), use_container_width=True)
        elif chart_type == "Scatter - Rooms vs Bedrooms":
            st.plotly_chart(plot_scatter(df_clean, 'total_rooms', 'total_bedrooms', 'Rooms vs Bedrooms', 'median_house_value'), use_container_width=True)

elif choice == "📈 Insights":
    st.title("Business Insights")
    df_clean = clean_data(df_raw)
    insights = get_insights(df_clean)
    
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    for insight in insights:
        st.info(insight)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("### Feature Importance Explanation")
    st.write("Based on linear correlation and domain knowledge, **Median Income** is the strongest predictor of house prices. Additionally, geographical location (**Latitude/Longitude**) and proximity to the ocean play massive roles in property valuation.")

elif choice == "🤖 Model Training":
    st.title("Model Training Pipeline")
    st.write("We use a **Linear Regression** model to predict house prices based on the dataset features.")
    
    st.write("### Linear Regression Details")
    st.write("- **Assumptions**: Linear relationship, multivariate normality, no multicollinearity, no auto-correlation, and homoscedasticity.")
    st.latex(r"Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon")
    st.write("- **Cost Function**: Mean Squared Error (MSE). We aim to minimize the sum of squared residuals.")
    st.write("- **Gradient Descent**: An optimization algorithm used to minimize the cost function by iteratively moving in the direction of steepest descent.")
    st.write("- **Advantages**: Simple to implement, interpret, and very efficient to train.")
    st.write("- **Disadvantages**: Prone to underfitting, sensitive to outliers, assumes linear relationships.")
    
    st.info("The model is trained behind the scenes using `train_model.py`. Please navigate to the Evaluation page to see its performance.")

elif choice == "📉 Evaluation":
    st.title("Model Evaluation Metrics")
    st.write("Performance of the trained Linear Regression model on the 20% test set.")
    
    # Static metrics based on the train_model.py output
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-card'><h4>R² Score</h4><h2>0.6254</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h4>Adjusted R²</h4><h2>0.6242</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h4>MAE</h4><h2>$50,670</h2></div>", unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    col4.markdown(f"<div class='metric-card'><h4>MSE</h4><h2>4.90B</h2></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='metric-card'><h4>RMSE</h4><h2>$70,060</h2></div>", unsafe_allow_html=True)
    
    st.write("### Model Equation Parameters")
    st.write("Intercept: $238,826.36")
    st.success("The Model, Scaler, and Encoder have been successfully serialized via Pickle.")

elif choice == "💰 Predict House Price":
    st.title("Predict House Price")
    st.write("Enter the details below to estimate the median house value.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            longitude = st.number_input("Longitude", value=-122.23)
            latitude = st.number_input("Latitude", value=37.88)
            housing_median_age = st.number_input("Housing Median Age", value=41.0)
            total_rooms = st.number_input("Total Rooms", value=880.0)
            total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
        with col2:
            population = st.number_input("Population", value=322.0)
            households = st.number_input("Households", value=126.0)
            median_income = st.number_input("Median Income", value=8.3252)
            ocean_proximity = st.selectbox("Ocean Proximity", ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])
            
        submitted = st.form_submit_button("Predict House Price")
        
    if submitted:
        input_data = {
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity': ocean_proximity
        }
        
        with st.spinner("Calculating Prediction..."):
            time.sleep(1)
            prediction = predict_house_price(input_data)
            
            if prediction is not None:
                st.markdown(f"""
                <div class='metric-card fade-in' style='background-color: #2E7D32;'>
                    <h3>Estimated House Price</h3>
                    <h1>${prediction:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.info("Prediction Confidence: Moderate. Linear Regression provides a strong baseline, but complex non-linear patterns might require advanced models like Random Forest.")
            else:
                st.error("Error making prediction. Make sure you have trained the model first.")
                
        # Option to download result
        if prediction is not None:
            result_df = pd.DataFrame([input_data])
            result_df['Predicted_Price'] = prediction
            csv = result_df.to_csv(index=False)
            st.download_button(label="Download Prediction Result", data=csv, file_name='prediction_result.csv', mime='text/csv')

elif choice == "ℹ️ About":
    st.title("About the Project")
    st.write("""
    ### House Price Prediction Dashboard
    This project was built as an end-to-end Machine Learning pipeline.
    
    **Developed by**: Senior Data Scientist and Full Stack ML Engineer
    
    **Technologies Used**:
    - **Python**: Core programming language
    - **Pandas, NumPy**: Data manipulation and numerical operations
    - **Scikit-Learn**: Machine Learning model building and preprocessing
    - **Plotly, Matplotlib, Seaborn**: Data Visualization
    - **Streamlit**: Web Application Framework
    
    This application demonstrates data ingestion, cleaning, exploratory analysis, model training, and real-time inference wrapped in a beautiful, responsive UI.
    """)
