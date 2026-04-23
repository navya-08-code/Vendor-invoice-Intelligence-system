import os
import joblib
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "..", "invoice_flagging", "models", "predict_flag_invoice.pkl")
SCALER_PATH = os.path.join(current_dir, "..", "invoice_flagging", "models", "scaler.pkl")

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load trained scaler.
    """
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
    
    Parameters
    -----------
    input_data : dict
    
    Returns
    ----------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    scaler = load_scaler()
    
    input_df = pd.DataFrame(input_data)
    
    features = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars"
    ]
    
    input_scaled = scaler.transform(input_df[features])
    input_df['Predicted_Flag'] = model.predict(input_scaled)
    
    return input_df

if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "invoice_quantity": [50, 10, 100],
        "invoice_dollars": [150.0, 50.0, 1000.0],
        "Freight": [10.0, 5.0, 50.0],
        "total_item_quantity": [50, 10, 100],
        "total_item_dollars": [150.0, 40.0, 1100.0]
    }
    
    prediction = predict_invoice_flag(sample_data)
    print(prediction)