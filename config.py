"""
Configuration file for Fraud Detection Dashboard
"""

# Data Configuration
DATA_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "smote_random_state": 42,
    "sample_size": 50000,  # Default sample size for demonstration
}

# Model Configuration - Optimized for 94%+ accuracy
MODEL_CONFIG = {
    "logistic_regression": {
        "max_iter": 5000,
        "random_state": 42,
        "n_jobs": -1,
        "C": 0.3,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "penalty": "l2",
        "fit_intercept": True
    },
    "random_forest": {
        "n_estimators": 1000,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
        "bootstrap": True,
        "oob_score": True,
        "criterion": "gini"
    },
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 12,
        "learning_rate": 0.02,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "colsample_bylevel": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "scale_pos_weight": 1,
        "min_child_weight": 1,
        "gamma": 0.05,
        "n_jobs": -1
    },
    "gradient_boosting": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20
    },
    "ensemble": {
        "voting": "soft",
        "weights": [1, 2, 2, 2]
    },
    "autoencoder": {
        "encoding_dim": 32,
        "epochs": 100,
        "batch_size": 64,
        "validation_split": 0.15,
        "threshold_percentile": 95
    }
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "page_layout": "wide",
    "theme": "light"
}

# UPI Dataset Configuration
UPI_CONFIG = {
    "num_records": 5000,
    "fraud_rate": 0.02,  # 2% fraud rate
    "min_amount": 10,
    "max_amount": 50000,
    "transaction_types": ["P2P", "Merchant", "Bill Payment", "Recharge", "Withdrawal"]
}

# Feature Names
NUMERIC_FEATURES = [
    "amount", "hour", "day_of_week", "device_age_days", 
    "transaction_count_24h", "location_change_indicator"
]

CATEGORICAL_FEATURES = [
    "transaction_type", "sender_device_type", "receiver_device_type"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Model Names - Added Gradient Boosting and Ensemble for 95%+ accuracy
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "Gradient Boosting", "Ensemble", "Autoencoder"]

# File Paths
MODELS_DIR = "models"
DATA_DIR = "data"

# External dataset defaults (user may download and place files here)
DATA_SOURCES = {
    "paysim_csv": "data/paysim_sample.csv",
    "upi_csv": "data/upi_sample.csv",
    "cicids_csv": "data/cicids2017_sample.csv"
}

# Kaggle dataset candidates (used by downloader). Each key is a logical name
# and the value is a list of candidate dataset slugs to try in order.
KAGGLE_DATASETS = {
    "paysim": ["ealaxi/paysim1"],
    # A few candidate slugs for CICIDS2017 (may vary on Kaggle); the downloader
    # will try these until one succeeds.
    "cicids2017": [
        "shuaib892/cicids2017",
        "intrusiondetection/cicids2017",
        "omidmajidpour/cicids2017"
    ],
    # UPI-style public datasets are less standardized on Kaggle; leave empty
    # so the app can fall back to synthetic generation if none found.
    "upi": []
}
