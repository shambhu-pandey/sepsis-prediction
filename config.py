"""
Configuration for Nexus Fraud Defense
"""

import os

MODELS_DIR = "models"
DATA_DIR = "data"

# --- Data pipeline ---
DATA_CONFIG = {
    "test_size": 0.20,
    "random_state": 42,
    "cv_folds": 5,
}

FRAUD_THRESHOLD = 0.25

# Columns that leak target information and must be removed
LEAKAGE_COLS = [
    "newbalanceOrig", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "isFlaggedFraud",
]

RAW_FEATURES_BY_DATASET = {
    # PaySim: keep all behavioral features
    "paysim": ["type", "amount", "oldbalanceOrg", "oldbalanceDest"],
    # BankSim: amount is a real behavioral signal (not leakage), step = day of sim
    "banksim": ["step", "age", "gender", "category", "amount"],
    # CICIDS: packet-level behavioral features (byte-rate cols cleaned of NaN/inf upstream)
    "cicids": ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
               "SYN Flag Count", "Packet Length Mean", "Average Packet Size"],
    # IEEE-CIS: dist1 is a genuine fraud signal (geo distance), not leakage
    "ieee": ["TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card4", "card6",
             "P_emaildomain", "DeviceType", "dist1"],
}

# --- Dataset metadata ---
DATASET_META = {
    "paysim": {
        "display_name": "PaySim  -  Payment Fraud",
        "short_name": "PaySim",
        "file": "data/paysim_clean.csv",
        "alt_files": ["data/paysim.csv"],
        "target_col": "isFraud",
        "sample_size": 250_000,
        "domain": "P2P / Mobile Payments",
        "description": "Synthetic mobile money transactions modeled on real African operator logs.",
    },
    "banksim": {
        "display_name": "BankSim  -  Retail Fraud",
        "short_name": "BankSim",
        "file": "data/banksim.csv",
        "alt_files": [],
        "target_col": "fraud",
        "sample_size": None,
        "domain": "Consumer Retail Banking",
        "description": "Simulated retail bank transactions with merchant categories.",
    },
    "cicids": {
        "display_name": "CICIDS2017  -  Network Intrusion",
        "short_name": "CICIDS",
        "file": "data/cicids.csv",
        "alt_files": ["data/cicids_clean.csv"],
        "target_col": "Label",
        "sample_size": 150_000,
        "domain": "Cyber-Security / Intrusion Detection",
        "description": "Realistic network traffic from the Canadian Institute for Cybersecurity.",
    },
    "ieee": {
        "display_name": "IEEE-CIS  -  E-Commerce Fraud",
        "short_name": "IEEE-CIS",
        "file": "data/ieee.csv",
        "alt_files": [],
        "target_col": "isFraud",
        "sample_size": None,
        "domain": "Online / E-Commerce Transactions",
        "description": "Vesta Corporation e-commerce transaction data.",
    },
}

# --- Model hyperparameters (constrained to prevent overfitting) ---
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Isolation Forest",
]

# --- Dataset-specific tuning (for realistic variation) ---
TUNING_CONFIG = {
    "paysim": {
        "rf": {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
        }
        ,
        "threshold": 0.005
    },
    "banksim": {
        "rf": {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
        }
        ,
        "threshold": 0.815
    },
    "cicids": {
        "rf": {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
        }
        ,
        "threshold": 0.005
    },
    "ieee": {
        "rf": {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
        }
        ,
        "threshold": 0.01
    },
}

MODEL_CONFIG = {
    "logistic_regression": {
        "max_iter": 2000,
        "random_state": 42,
        "C": 1.0,
        "solver": "lbfgs",
        "penalty": "l2",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": 1,
    },
    "xgboost": {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "min_child_weight": 5,
        "gamma": 0.2,
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": 1,
    },
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": "auto",
        "random_state": 42,
        "n_jobs": 1,
        "max_samples": 0.8,
    },
}

# --- Default fraud demo inputs (pre-filled) ---
FRAUD_EXAMPLES = {
    "paysim": {
        "type": "TRANSFER",
        "amount": 2754740.54,
        "oldbalanceOrg": 2754740.54,
        "oldbalanceDest": 0.0,
    },
    "banksim": {
        "step": 1,
        "amount": 25000.0,
        "category": "es_travel",
        "age": "U",
        "gender": "E",
    },
    "cicids": {
        "Destination Port": 80,
        "Flow Duration": 10000000.0,
        "Total Fwd Packets": 50000,
        "Total Backward Packets": 40000,
        "Total Length of Fwd Packets": 5000000.0,
        "Total Length of Bwd Packets": 5000000.0,
        "Flow Bytes/s": 50000000.0,
        "Flow Packets/s": 50000.0,
        "Packet Length Mean": 4000.0,
        "Average Packet Size": 4500.0,
        "SYN Flag Count": 200,
    },
    "ieee": {
        "TransactionDT": 86400,
        "TransactionAmt": 1500.0,
        "ProductCD": "C",
        "card1": 10000,
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "anonymous.com",
        "DeviceType": "mobile",
        "dist1": 500.0,
    },
}

NORMAL_EXAMPLES = {
    "paysim": {
        "type": "PAYMENT",
        "amount": 5000.0,
        "oldbalanceOrg": 20000.0,
        "oldbalanceDest": 10000.0,
    },
    "banksim": {
        "step": 1,
        "amount": 25.0,
        "category": "es_food",
        "age": "3",
        "gender": "M",
    },
    "cicids": {
        "Destination Port": 443,
        "Flow Duration": 500.0,
        "Total Fwd Packets": 5,
        "Total Backward Packets": 3,
        "Total Length of Fwd Packets": 500.0,
        "Total Length of Bwd Packets": 1500.0,
        "Flow Bytes/s": 200.0,
        "Flow Packets/s": 10.0,
        "Packet Length Mean": 120.0,
        "Average Packet Size": 130.0,
        "SYN Flag Count": 0,
    },
    "ieee": {
        "TransactionDT": 86400,
        "TransactionAmt": 45.0,
        "ProductCD": "W",
        "card1": 15000,
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "dist1": 5.0,
    },
}

DEFAULT_FRAUD_INPUTS = FRAUD_EXAMPLES

# --- UI ---
DASHBOARD_CONFIG = {
    "page_title": "Nexus Fraud Defense",
    "page_icon": "",
    "layout": "wide",
}

# --- Legacy compat ---
NUMERIC_FEATURES = ["amount"]
CATEGORICAL_FEATURES = ["transaction_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODELS = MODEL_NAMES
PAYSIM_LEAKAGE_COLS = LEAKAGE_COLS

UPI_CONFIG = {
    "num_records": 5000, "fraud_rate": 0.02,
    "min_amount": 10, "max_amount": 50000,
    "transaction_types": ["P2P", "Merchant", "Bill Payment", "Recharge", "Withdrawal"],
}
