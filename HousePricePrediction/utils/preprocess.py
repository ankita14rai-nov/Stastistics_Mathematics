import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

def load_data(filepath='housing.csv'):
    """Loads the housing dataset."""
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Cleans the dataframe:
    - Removes duplicates
    - Fills missing values (total_bedrooms with median)
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    if df['total_bedrooms'].isnull().sum() > 0:
        median_bedrooms = df['total_bedrooms'].median()
        df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
        
    return df

def feature_engineering(df):
    """
    Creates new features based on existing ones.
    """
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    return df

def preprocess_and_split(df, target_col='median_house_value', test_size=0.2, random_state=42):
    """
    Splits the data and fits/transforms the numerical and categorical features.
    Returns X_train, X_test, y_train, y_test, scaler, encoder.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Separate numerical and categorical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])
    
    X_train_num_df = pd.DataFrame(X_train_num, columns=num_cols, index=X_train.index)
    X_test_num_df = pd.DataFrame(X_test_num, columns=num_cols, index=X_test.index)
    
    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
    
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train.index)
    X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_feature_names, index=X_test.index)
    
    # Combine numerical and categorical features
    X_train_processed = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
    X_test_processed = pd.concat([X_test_num_df, X_test_cat_df], axis=1)
    
    return X_train_processed, X_test_processed, y_train, y_test, scaler, encoder

