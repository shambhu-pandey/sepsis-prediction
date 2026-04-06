"""
Explainability Module — SHAP, LIME, and feature importance visualizations.
Addresses the Problem Statement requirement for transparent, user-friendly AI explanations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

# Lazy imports to avoid crashing if optional deps missing
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

from modules.model_trainer import _extract_estimator, _predict_scores


# ───────────────────────────── SHAP ─────────────────────────────

def generate_shap_explanation(model_wrapper, X_train_sample, X_instance, feature_names):
    """
    Render a SHAP waterfall or summary-bar plot for a single transaction.
    Works for tree-based models (RF / XGBoost) and falls back to KernelExplainer.
    """
    if not _SHAP_AVAILABLE:
        st.warning("SHAP is not installed. Run `pip install shap` to enable this feature.")
        return False

    estimator = _extract_estimator(model_wrapper)
    st.markdown("#### 🔵 SHAP — Additive Feature Contributions")
    st.markdown("Each bar shows how much a feature pushed the fraud probability **up (red) or down (blue)** from the baseline average.")

    try:
        type_str = str(type(estimator)).lower()
        if 'forest' in type_str or 'xgb' in type_str or 'gradient' in type_str:
            explainer = shap.TreeExplainer(estimator)
            shap_output = explainer(X_instance)

            # Handle multi-class output shape [samples, features, classes]
            if hasattr(shap_output, 'values') and len(np.array(shap_output.values).shape) == 3:
                val_to_plot = shap_output[:, :, 1]
            else:
                val_to_plot = shap_output

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(val_to_plot[0], show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            # KernelExplainer fallback for logistic / ensemble
            bg = shap.sample(X_train_sample, min(50, len(X_train_sample)))
            pred_fn = lambda x: _predict_scores(model_wrapper, pd.DataFrame(x, columns=feature_names))
            explainer = shap.KernelExplainer(pred_fn, bg)
            shap_vals = explainer.shap_values(X_instance)

            fig, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_vals, X_instance, feature_names=feature_names,
                              plot_type="bar", show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        return True
    except Exception as exc:
        st.warning(f"SHAP rendering failed: {exc}")
        return False


def plot_shap_summary(model_wrapper, X_sample, feature_names):
    """
    Render a SHAP beeswarm summary plot across a dataset sample.
    Useful for understanding global feature importance.
    """
    if not _SHAP_AVAILABLE:
        st.warning("SHAP not installed.")
        return False

    estimator = _extract_estimator(model_wrapper)
    try:
        type_str = str(type(estimator)).lower()
        if 'forest' in type_str or 'xgb' in type_str or 'gradient' in type_str:
            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # class=1 for binary
        else:
            bg = shap.sample(X_sample, min(50, len(X_sample)))
            pred_fn = lambda x: _predict_scores(model_wrapper, pd.DataFrame(x, columns=feature_names))
            explainer = shap.KernelExplainer(pred_fn, bg)
            shap_vals = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return True
    except Exception as exc:
        st.warning(f"SHAP summary failed: {exc}")
        return False


# ───────────────────────────── LIME ─────────────────────────────

def generate_lime_explanation(model_wrapper, X_train_sample, X_instance, feature_names):
    """
    Render a LIME local surrogate bar chart explaining an individual prediction.
    """
    if not _LIME_AVAILABLE:
        st.warning("LIME is not installed. Run `pip install lime` to enable this feature.")
        return False

    st.markdown("#### 🟠 LIME — Local Surrogate Model")
    st.markdown("Simulates thousands of slight variations around this transaction to show what *tipped* the model's decision.")

    def predict_proba_wrapper(x):
        try:
            probs_1 = _predict_scores(model_wrapper, pd.DataFrame(x, columns=feature_names))
            return np.vstack((1 - probs_1, probs_1)).T
        except Exception:
            return np.zeros((x.shape[0], 2))

    try:
        sample_n = min(500, len(X_train_sample))
        X_bg = X_train_sample.sample(sample_n, random_state=42) if len(X_train_sample) > sample_n else X_train_sample

        explainer = LimeTabularExplainer(
            training_data=X_bg.values,
            feature_names=feature_names,
            class_names=['Legitimate', 'Fraud'],
            mode='classification',
            random_state=42
        )
        exp = explainer.explain_instance(
            data_row=X_instance.iloc[0].values,
            predict_fn=predict_proba_wrapper,
            num_features=10
        )
        fig = exp.as_pyplot_figure()
        plt.title("Feature Attribution — What Drove the Fraud Score?", pad=14)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return True
    except Exception as exc:
        st.warning(f"LIME rendering failed: {exc}")
        return False


# ───────────────────────── Feature Importance ───────────────────

def plot_feature_importance(importance_dict, title="Feature Importance", top_n=15):
    """
    Render a horizontal bar chart of feature importances.
    Works with any dict of {feature_name: importance_value}.
    """
    if not importance_dict:
        st.info("No feature importance data available.")
        return

    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [i[0] for i in items]
    values = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    bars = ax.barh(names[::-1], values[::-1],
                   color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(names))))
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────── Unified Entry-Point ─────────────────

def explain_prediction(model_wrapper, X_train_sample, X_instance, feature_names,
                       use_shap=True, use_lime=True):
    """
    Unified convenience wrapper that renders both SHAP and LIME panels
    inside the active Streamlit context.
    """
    explained = False
    if use_shap:
        explained = generate_shap_explanation(model_wrapper, X_train_sample, X_instance, feature_names)
        st.markdown("---")
    if use_lime:
        explained = generate_lime_explanation(model_wrapper, X_train_sample, X_instance, feature_names) or explained
    return explained
