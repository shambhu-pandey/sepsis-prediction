"""
Fraud Detection Dashboard Modules
"""

from .data_loader import (
    generate_upi_dataset,
    load_upi_csv,
    load_cicids2017_csv,
    load_dataset,
    preprocess_data,
    split_data,
    validate_and_prepare_transaction
)

from .model_trainer import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_autoencoder,
    evaluate_model,
    get_feature_importance,
    predict_fraud,
    train_all_models
)

from .explainability import (
    generate_shap_explanation,
    generate_lime_explanation,
    plot_feature_importance,
    plot_shap_summary,
    explain_prediction
)

from .digital_twin import (
    DigitalTwinSimulator,
    get_digital_twin_dashboard_data
)

from .utils import (
    save_model,
    load_latest_model,
    generate_risk_score,
    validate_transaction_data,
    generate_sample_transaction
)

__all__ = [
    "generate_upi_dataset",
    "load_upi_csv",
    "load_cicids2017_csv",
    "load_dataset",
    "preprocess_data",
    "split_data",
    "validate_and_prepare_transaction",
    "train_logistic_regression",
    "train_random_forest",
    "train_xgboost",
    "train_autoencoder",
    "evaluate_model",
    "get_feature_importance",
    "predict_fraud",
    "train_all_models",
    "generate_shap_explanation",
    "generate_lime_explanation",
    "plot_feature_importance",
    "plot_shap_summary",
    "explain_prediction",
    "DigitalTwinSimulator",
    "get_digital_twin_dashboard_data",
    "save_model",
    "load_latest_model",
    "generate_risk_score",
    "validate_transaction_data",
    "generate_sample_transaction"
]
