"""
Model Training and Evaluation Module
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from config import MODEL_CONFIG
from modules.utils import save_model, calculate_confusion_matrix_metrics


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    model = LogisticRegression(**MODEL_CONFIG["logistic_regression"])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    model = RandomForestClassifier(**MODEL_CONFIG["random_forest"])
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    model = xgb.XGBClassifier(**MODEL_CONFIG["xgboost"])
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """
    Train Gradient Boosting model for higher accuracy.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    model = GradientBoostingClassifier(**MODEL_CONFIG["gradient_boosting"])
    model.fit(X_train, y_train)
    return model


def train_ensemble(X_train, y_train):
    """
    Train Ensemble Voting Classifier for maximum accuracy.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained ensemble model
    """
    # Create base classifiers
    lr = LogisticRegression(**MODEL_CONFIG["logistic_regression"])
    rf = RandomForestClassifier(**MODEL_CONFIG["random_forest"])
    xgb_model = xgb.XGBClassifier(**MODEL_CONFIG["xgboost"])
    gb = GradientBoostingClassifier(**MODEL_CONFIG["gradient_boosting"])
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('xgb', xgb_model),
            ('gb', gb)
        ],
        **MODEL_CONFIG["ensemble"]
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble


def train_autoencoder(X_train, y_train, X_test=None):
    """
    Train Autoencoder for anomaly detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
    
    Returns:
        dict: model, threshold, reconstruction_errors
    """
    config = MODEL_CONFIG["autoencoder"]
    
    # Build autoencoder
    input_dim = X_train.shape[1]
    encoding_dim = config["encoding_dim"]
    
    # Encoder
    input_layer = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    
    # Decoder
    decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train on normal transactions only
    X_normal = X_train[y_train == 0]
    
    autoencoder.fit(
        X_normal, X_normal,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=config["validation_split"],
        verbose=0
    )
    
    # Calculate reconstruction error on training set
    X_pred = autoencoder.predict(X_train, verbose=0)
    mse = np.mean(np.square(X_train - X_pred), axis=1)
    
    # Set threshold
    threshold = np.percentile(mse, config["threshold_percentile"])
    
    return {
        "model": autoencoder,
        "threshold": threshold,
        "reconstruction_errors": mse
    }


def evaluate_model(model, X_test, y_test, model_type="classification"):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_type: Type of model ("classification" or "autoencoder")
    
    Returns:
        dict: Metrics
    """
    if model_type == "autoencoder":
        # For autoencoder, use reconstruction error
        X_pred = model["model"].predict(X_test, verbose=0)
        mse = np.mean(np.square(X_test - X_pred), axis=1)
        y_pred = (mse > model["threshold"]).astype(int)
    else:
        # For classification models
        y_pred = model.predict(X_test)
        
        # Get probability estimates if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        if model_type == "autoencoder":
            roc_auc = roc_auc_score(y_test, mse)
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0.0
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }
    
    return metrics, y_pred, y_pred_proba if model_type != "autoencoder" else mse


def get_feature_importance(model, feature_names, model_name="Random Forest"):
    """
    Get feature importance from model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of model
    
    Returns:
        dict: Feature importance mapping
    """
    importance_dict = {}
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            importance_dict[name] = importance
    
    elif hasattr(model, 'coef_'):
        # Linear models
        coef = np.abs(model.coef_[0])
        for name, coef_val in zip(feature_names, coef):
            importance_dict[name] = coef_val
    
    # Normalize to sum to 1
    if importance_dict:
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict


def predict_fraud(model, X_input, model_type="classification"):
    """
    Predict fraud for input data.
    
    Args:
        model: Trained model
        X_input: Input features
        model_type: Type of model
    
    Returns:
        tuple: (prediction, probability)
    """
    if model_type == "autoencoder":
        X_pred = model["model"].predict(X_input, verbose=0)
        mse = np.mean(np.square(X_input - X_pred), axis=1)
        prediction = (mse > model["threshold"]).astype(int)[0]
        probability = float(mse[0] / model["threshold"])
        probability = min(probability, 1.0)  # Cap at 1.0
    else:
        prediction = model.predict(X_input)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X_input)[0, 1]
        else:
            probability = float(prediction)
    
    return int(prediction), float(probability)


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all models and return results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Trained models and results
    """
    results = {}
    models = {}
    
    # Logistic Regression
    try:
        lr_model = train_logistic_regression(X_train, y_train)
        lr_metrics, lr_pred, lr_proba = evaluate_model(lr_model, X_test, y_test)
        results["Logistic Regression"] = lr_metrics
        models["Logistic Regression"] = {
            "model": lr_model,
            "predictions": lr_pred,
            "probabilities": lr_proba
        }
    except Exception as e:
        st.warning(f"Error training Logistic Regression: {e}")
    
    # Random Forest
    try:
        rf_model = train_random_forest(X_train, y_train)
        rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test)
        results["Random Forest"] = rf_metrics
        models["Random Forest"] = {
            "model": rf_model,
            "predictions": rf_pred,
            "probabilities": rf_proba
        }
    except Exception as e:
        st.warning(f"Error training Random Forest: {e}")
    
    # XGBoost
    try:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_metrics, xgb_pred, xgb_proba = evaluate_model(xgb_model, X_test, y_test)
        results["XGBoost"] = xgb_metrics
        models["XGBoost"] = {
            "model": xgb_model,
            "predictions": xgb_pred,
            "probabilities": xgb_proba
        }
    except Exception as e:
        st.warning(f"Error training XGBoost: {e}")
    
    # Gradient Boosting - for 95%+ accuracy
    try:
        gb_model = train_gradient_boosting(X_train, y_train)
        gb_metrics, gb_pred, gb_proba = evaluate_model(gb_model, X_test, y_test)
        results["Gradient Boosting"] = gb_metrics
        models["Gradient Boosting"] = {"model": gb_model, "predictions": gb_pred, "probabilities": gb_proba}
    except Exception as e:
        st.warning(f"Error training Gradient Boosting: {e}")
    
    # Ensemble - for 95%+ accuracy
    try:
        ens_model = train_ensemble(X_train, y_train)
        ens_metrics, ens_pred, ens_proba = evaluate_model(ens_model, X_test, y_test)
        results["Ensemble"] = ens_metrics
        models["Ensemble"] = {"model": ens_model, "predictions": ens_pred, "probabilities": ens_proba}
    except Exception as e:
        st.warning(f"Error training Ensemble: {e}")
    
    # Autoencoder
    try:
        ae_model = train_autoencoder(X_train, y_train, X_test)
        ae_metrics, ae_pred, ae_reconstructon_errors = evaluate_model(
            ae_model, X_test, y_test, model_type="autoencoder"
        )
        results["Autoencoder"] = ae_metrics
        models["Autoencoder"] = {
            "model": ae_model,
            "predictions": ae_pred,
            "reconstruction_errors": ae_reconstructon_errors
        }
    except Exception as e:
        st.warning(f"Error training Autoencoder: {e}")
    
    return results, models
