"""
Explainability Module using SHAP and LIME
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from modules.model_trainer import get_feature_importance


def generate_shap_explanation(model, X_data, X_sample, feature_names, model_type="classification", max_display=10):
    """
    Generate SHAP explanation for model predictions.
    
    Args:
        model: Trained model
        X_data: Background data for SHAP
        X_sample: Sample to explain
        feature_names: Feature names
        model_type: Type of model
        max_display: Maximum features to display
    
    Returns:
        dict: SHAP values and explanation
    """
    try:
        if model_type == "autoencoder":
            # For autoencoder, just return feature norms
            sample_array = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
            reconstruction = model["model"].predict(sample_array, verbose=0)
            errors = np.abs(sample_array - reconstruction)[0]
            
            feature_contributions = {}
            for name, error in zip(feature_names, errors):
                feature_contributions[name] = float(error)
            
            return {
                "shap_values": feature_contributions,
                "type": "reconstruction_error",
                "base_value": float(np.mean(errors))
            }
        
        # For tree-based models, use TreeExplainer
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        else:
            # For other models, use KernelExplainer
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                shap.sample(X_data, min(100, len(X_data)))
            )
        
        # Get SHAP values
        if isinstance(X_sample, pd.DataFrame):
            X_sample_array = X_sample.values
        else:
            X_sample_array = X_sample
        
        shap_values = explainer.shap_values(X_sample_array)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification returns [values_class_0, values_class_1]
            shap_values = shap_values[1]
        
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        # Create feature contribution dict
        feature_contributions = {}
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        for name, value in zip(feature_names, shap_values):
            feature_contributions[name] = float(value)
        
        # Sort by absolute value
        feature_contributions = dict(sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_display])
        
        return {
            "shap_values": feature_contributions,
            "base_value": float(base_value),
            "type": "shap_explanation"
        }
    
    except Exception as e:
        st.warning(f"Error generating SHAP explanation: {e}")
        return None


def generate_lime_explanation(model, X_train, X_sample, feature_names, model_type="classification", num_features=10):
    """
    Generate LIME explanation for model predictions.
    
    Args:
        model: Trained model
        X_train: Training data for reference
        X_sample: Sample to explain
        feature_names: Feature names
        model_type: Type of model
        num_features: Number of features to display
    
    Returns:
        dict: LIME explanation
    """
    try:
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
        
        if isinstance(X_sample, pd.DataFrame):
            X_sample_array = X_sample.values[0]
        else:
            X_sample_array = X_sample[0]
        
        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_array,
            feature_names=feature_names,
            class_names=['Not Fraud', 'Fraud'],
            mode='classification',
            verbose=False
        )
        
        # Create prediction function
        if hasattr(model, 'predict_proba'):
            pred_fn = model.predict_proba
        else:
            pred_fn = lambda x: np.array([1 - model.predict(x), model.predict(x)]).T
        
        # Explain prediction
        exp = explainer.explain_instance(
            X_sample_array,
            pred_fn,
            num_features=num_features
        )
        
        # Extract feature contributions
        feature_contributions = {}
        for feature_name, weight in exp.as_list():
            feature_contributions[feature_name] = weight
        
        return {
            "lime_explanation": feature_contributions,
            "type": "lime_explanation"
        }
    
    except Exception as e:
        st.warning(f"Error generating LIME explanation: {e}")
        return None


def plot_feature_importance(model, feature_names, model_name="Model", top_n=10):
    """
    Plot feature importance.
    
    Args:
        model: Trained model
        feature_names: Feature names
        model_name: Name of model
        top_n: Number of top features to show
    
    Returns:
        matplotlib figure
    """
    importance_dict = get_feature_importance(model, feature_names, model_name)
    
    # Get top N features
    top_features = dict(list(importance_dict.items())[:top_n])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(top_features.keys())
    values = list(top_features.values())
    
    ax.barh(names, values, color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Feature Importance - {model_name}')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_shap_summary(model, X_data, feature_names, model_type="classification"):
    """
    Plot SHAP summary plot.
    
    Args:
        model: Trained model
        X_data: Data to explain
        feature_names: Feature names
        model_type: Type of model
    
    Returns:
        matplotlib figure or None
    """
    try:
        if model_type == "autoencoder":
            return None
        
        if isinstance(X_data, pd.DataFrame):
            X_array = X_data.values
        else:
            X_array = X_data
        
        # Create explainer based on model type
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                shap.sample(X_data, min(50, len(X_data)))
            )
        
        shap_values = explainer.shap_values(X_array[:50])  # Limit for speed
        
        # Handle list output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_array[:50], feature_names=feature_names, show=False)
        
        return fig
    
    except Exception as e:
        st.warning(f"Error creating SHAP plot: {e}")
        return None


def explain_prediction(model, X_sample, feature_names, original_data, model_type="classification", use_lime=False):
    """
    Generate comprehensive explanation for a prediction.
    
    Args:
        model: Trained model
        X_sample: Processed sample to explain
        feature_names: Feature names
        original_data: Original transaction data
        model_type: Type of model
        use_lime: Whether to use LIME
    
    Returns:
        dict: Comprehensive explanation
    """
    from modules.model_trainer import predict_fraud
    
    # Get prediction
    prediction, probability = predict_fraud(model, X_sample, model_type)
    
    explanation = {
        "prediction": prediction,
        "probability": probability,
        "features": {}
    }
    
    # Add feature values
    if isinstance(X_sample, pd.DataFrame):
        for feature_name in feature_names:
            if feature_name in X_sample.columns:
                explanation["features"][feature_name] = float(X_sample[feature_name].values[0])
    
    # Generate SHAP explanation (if possible)
    try:
        shap_exp = generate_shap_explanation(model, pd.DataFrame(np.zeros((10, len(feature_names))), columns=feature_names), X_sample, feature_names, model_type)
        if shap_exp:
            explanation["shap"] = shap_exp
    except:
        pass
    
    # Add anomaly indicators based on features
    anomalies = []
    
    # Check for unusual amount
    if "amount" in X_sample.columns:
        amount = X_sample["amount"].values[0]
        if amount > 40000:
            anomalies.append(f"Unusually high amount: ₹{amount:,.2f}")
    
    # Check for new device
    if "device_age_days" in X_sample.columns:
        device_age = X_sample["device_age_days"].values[0]
        if device_age < 30:
            anomalies.append(f"New device (age: {device_age:.0f} days)")
    
    # Check for location change
    if "location_change_indicator" in X_sample.columns:
        loc_change = X_sample["location_change_indicator"].values[0]
        if loc_change > 0.5:
            anomalies.append("Location change detected")
    
    # Check for unusual time
    if "hour" in X_sample.columns:
        hour = X_sample["hour"].values[0]
        if hour < 5 or hour > 23:
            anomalies.append(f"Unusual transaction time: {hour}:00")
    
    # Check for high transaction count
    if "transaction_count_24h" in X_sample.columns:
        txn_count = X_sample["transaction_count_24h"].values[0]
        if txn_count > 15:
            anomalies.append(f"High transaction count in 24h: {txn_count:.0f}")
    
    explanation["anomalies"] = anomalies
    
    return explanation
