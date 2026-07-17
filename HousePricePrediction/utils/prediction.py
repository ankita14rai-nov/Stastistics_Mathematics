import pickle
import pandas as pd
import numpy as np

def load_artifacts():
    """Loads the saved model, scaler, and encoder."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, scaler, encoder
    except Exception as e:
        return None, None, None

def predict_house_price(input_data):
    """
    Predicts the house price given input dictionary.
    """
    model, scaler, encoder = load_artifacts()
    if model is None:
        return None

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Separate numerical and categorical
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Scale
    num_scaled = scaler.transform(df[num_cols])
    
    # Encode
    cat_encoded = encoder.transform(df[cat_cols])
    
    # Combine
    X_processed = np.concatenate((num_scaled, cat_encoded), axis=1)
    
    # Predict
    prediction = model.predict(X_processed)
    return prediction[0]
