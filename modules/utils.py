"""
Utility functions for Fraud Detection Dashboard
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from config import MODELS_DIR


def ensure_models_directory():
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, model_name):
    ensure_models_directory()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"{model_name}_{ts}.pkl")
    joblib.dump(model, path)
    return path


def load_latest_model(model_name):
    ensure_models_directory()
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith(model_name)]
    if not files:
        return None
    return joblib.load(os.path.join(MODELS_DIR, sorted(files)[-1]))


def generate_risk_score(fraud_probability, feature_importance_dict=None):
    score = int(fraud_probability * 100)
    if score < 20:
        level, desc = "LOW", "Transaction appears safe"
    elif score < 50:
        level, desc = "MEDIUM", "Transaction warrants review"
    elif score < 80:
        level, desc = "HIGH", "Transaction likely fraudulent"
    else:
        level, desc = "VERY HIGH", "Strongly flagged as fraud"
    return {"risk_score": score, "risk_level": level,
            "description": desc, "fraud_probability": fraud_probability}


def validate_transaction_data(tx):
    for f in ["amount"]:
        if f not in tx:
            return False, f"Missing: {f}"
    if not isinstance(tx.get("amount"), (int, float)) or tx["amount"] <= 0:
        return False, "Amount must be positive"
    return True, "Valid"


def generate_sample_transaction():
    return {
        "amount": float(np.random.choice([500, 1000, 5000, 10000, 25000])),
        "hour": int(np.random.randint(0, 24)),
        "day_of_week": int(np.random.randint(0, 7)),
    }


def calculate_confusion_matrix_metrics(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "TPR": tp / (tp + fn + 1e-10), "FPR": fp / (fp + tn + 1e-10)}
