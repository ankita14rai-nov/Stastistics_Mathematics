# House Price Prediction Dashboard

## Project Overview
The **House Price Prediction Dashboard** is an end-to-end Machine Learning project designed to predict housing prices using the California Housing Dataset. The project includes data cleaning, exploratory data analysis (EDA), model training, evaluation, and a fully interactive web application built with Streamlit.

## Features
- **Interactive Data Exploration**: View dataset statistics, handle missing values, and analyze distributions.
- **Automated Business Insights**: Automatically generate and display insights related to correlations, most expensive locations, and more.
- **Real-time Price Prediction**: Input custom feature values to instantly get an estimated house price.
- **Beautiful Visualizations**: Professional interactive charts built using Plotly.
- **Dark Theme UI**: A sleek, modern, and responsive user interface with CSS styling and micro-animations.

## Dataset
This project uses the **California Housing Dataset**. It contains information such as:
- Longitude & Latitude
- Housing Median Age
- Total Rooms & Total Bedrooms
- Population & Households
- Median Income
- Ocean Proximity
- **Target Variable**: Median House Value

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Scikit-Learn**: Machine Learning model building and data preprocessing.
- **Plotly, Matplotlib, Seaborn**: Data Visualization.
- **Streamlit**: Web Application Framework for the interactive dashboard.

## Installation
Follow these steps to run the project locally:

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd HousePricePrediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   Execute the training script to generate the required model and preprocessing artifacts:
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Model Performance
The baseline model is a **Linear Regression** algorithm, yielding the following approximate metrics on the test set:
- **R² Score**: 0.6254
- **Adjusted R²**: 0.6242
- **MAE**: ~$50,670
- **MSE**: ~4.90B
- **RMSE**: ~$70,060

## Screenshots Placeholder
*(Add screenshots of the Home, EDA, and Prediction pages here)*
- `![Home Page](images/home.png)`
- `![EDA Page](images/eda.png)`
- `![Prediction Page](images/prediction.png)`

## Future Improvements
- Implement advanced models like **Random Forest** or **XGBoost** to capture non-linear relationships and improve R² score.
- Add hyperparameter tuning capabilities via the dashboard.
- Provide geographical maps for better visualization of price distributions.
- Deploy the application on a cloud platform (e.g., AWS, Heroku, or Streamlit Cloud).
