"""
Utility functions for Fraud Detection Dashboard
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from config import MODELS_DIR


def ensure_models_directory():
    """Ensure models directory exists."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)


def save_model(model, model_name):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name of the model
    
    Returns:
        str: Path to saved model
    """
    ensure_models_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.pkl")
    joblib.dump(model, model_path)
    return model_path


def load_latest_model(model_name):
    """
    Load the latest trained model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Loaded model or None if not found
    """
    ensure_models_directory()
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(model_name)]
    if not model_files:
        return None
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(MODELS_DIR, latest_model)
    return joblib.load(model_path)


def get_transaction_time_features(datetime_obj):
    """
    Extract time-based features from datetime.
    
    Args:
        datetime_obj: datetime object
    
    Returns:
        dict: Time features
    """
    return {
        "hour": datetime_obj.hour,
        "day_of_week": datetime_obj.weekday(),
        "day": datetime_obj.day,
        "month": datetime_obj.month
    }


def calculate_device_age_days(device_registration_date):
    """
    Calculate device age in days.
    
    Args:
        device_registration_date: datetime of device registration
    
    Returns:
        int: Number of days
    """
    return (datetime.now() - device_registration_date).days


def generate_risk_score(fraud_probability, feature_importance_dict=None):
    """
    Generate a risk score between 0-100 based on fraud probability.
    
    Args:
        fraud_probability: Probability of fraud (0-1)
        feature_importance_dict: Optional dict of feature contributions
    
    Returns:
        dict: Risk score details
    """
    risk_score = int(fraud_probability * 100)
    
    if risk_score < 20:
        risk_level = "🟢 LOW"
        description = "Transaction appears safe"
    elif risk_score < 50:
        risk_level = "🟡 MEDIUM"
        description = "Transaction warrants review"
    elif risk_score < 80:
        risk_level = "HIGH"
        description = "Transaction likely fraudulent"
    else:
        risk_level = "VERY HIGH"
        description = "Transaction strongly flagged as fraud"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "description": description,
        "fraud_probability": fraud_probability
    }


def validate_transaction_data(transaction_dict):
    """
    Validate transaction data input.
    
    Args:
        transaction_dict: Dictionary containing transaction details
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    required_fields = ["amount", "sender", "receiver", "device_id", "time", "transaction_type"]
    
    for field in required_fields:
        if field not in transaction_dict or transaction_dict[field] is None:
            return False, f"Missing required field: {field}"
    
    if not isinstance(transaction_dict.get("amount"), (int, float)) or transaction_dict["amount"] <= 0:
        return False, "Amount must be a positive number"
    
    if len(str(transaction_dict.get("sender", ""))) < 3:
        return False, "Invalid sender ID"
    
    if len(str(transaction_dict.get("receiver", ""))) < 3:
        return False, "Invalid receiver ID"
    
    return True, "Valid transaction data"


def format_metrics_table(metrics_dict):
    """
    Format metrics dictionary into a clean display format.
    
    Args:
        metrics_dict: Dictionary of metrics
    
    Returns:
        pd.DataFrame: Formatted metrics
    """
    df = pd.DataFrame(metrics_dict, index=[0])
    df = df.T
    df.columns = ["Value"]
    df["Value"] = df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    return df


def calculate_confusion_matrix_metrics(y_true, y_pred):
    """
    Calculate confusion matrix and related metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: TP, TN, FP, FN, TPR, FPR
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr
    }


def generate_sample_transaction():
    """
    Generate a sample transaction for demonstration.
    
    Returns:
        dict: Sample transaction
    """
    amounts = [500, 1000, 2000, 5000, 10000, 25000, 50000]
    types = ["P2P", "Merchant", "Bill Payment", "Recharge", "Withdrawal"]
    device_types = ["Android", "iOS", "Web"]
    
    return {
        "amount": np.random.choice(amounts),
        "sender": f"USER{np.random.randint(1000, 9999)}",
        "receiver": f"USER{np.random.randint(1000, 9999)}",
        "device_id": f"DEV{np.random.randint(10000, 99999)}",
        # return a datetime object for easier feature extraction
        "time": datetime.now(),
        "transaction_type": np.random.choice(types),
        "device_type": np.random.choice(device_types),
        "location": f"Location_{np.random.randint(1, 100)}",
        "transaction_count_24h": np.random.randint(1, 20),
        "location_change_indicator": np.random.choice([0, 1], p=[0.9, 0.1]),
        # include device age in days for risk assessment
        "device_age_days": float(np.random.uniform(0, 730))
    }
