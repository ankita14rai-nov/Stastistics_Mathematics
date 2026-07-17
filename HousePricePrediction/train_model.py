import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.preprocess import load_data, clean_data, feature_engineering, preprocess_and_split
import os

def main():
    print("Starting Model Training Pipeline...")
    
    # 1. Load Data
    data_path = 'housing.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return
        
    df = load_data(data_path)
    print(f"Data Loaded. Shape: {df.shape}")
    
    # 2. Clean Data
    df_clean = clean_data(df)
    print(f"Data Cleaned. Shape: {df_clean.shape}")
    
    # 3. Feature Engineering (Optional based on performance, skipping to keep simple as requested or we can add it)
    # Let's apply feature engineering as it usually helps Linear Regression on this dataset
    # df_engineered = feature_engineering(df_clean)
    df_engineered = df_clean # Using raw features as per standard
    
    # 4. Preprocess and Split
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_and_split(df_engineered)
    print(f"Data Split & Preprocessed. Training set shape: {X_train.shape}")
    
    # 5. Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model Trained successfully.")
    
    # 6. Evaluate Model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Adjusted R2
    n = X_test.shape[0]
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print("\n--- Model Evaluation ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"Adjusted R2 Score: {adj_r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    print("\n--- Model Parameters ---")
    print(f"Intercept: {model.intercept_:.2f}")
    
    # 7. Save Model and Preprocessing Objects
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
        
    print("\nModel, Scaler, and Encoder saved successfully using Pickle.")

if __name__ == "__main__":
    main()
