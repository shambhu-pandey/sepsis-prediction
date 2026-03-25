# Developer Guide

## Architecture Overview

### Project Structure
```
FRAUD_DETECTION/
├── app.py                    # Main Streamlit UI layer
├── config.py                 # Configuration management
├── modules/                  # Business logic layer
│   ├── data_loader.py       # External interfaces
│   ├── model_trainer.py     # ML algorithms
│   ├── explainability.py    # Interpretability
│   ├── digital_twin.py      # Simulation
│   └── utils.py             # Helpers
├── data/                     # Data artifacts
├── models/                   # Model artifacts
└── tests/                    # Unit tests (future)
```

### Design Pattern: Modular Architecture
- **Separation of Concerns**: Each module has single responsibility
- **Reusability**: Modules can be used independently
- **Testability**: Easy to test individual components
- **Scalability**: Easy to add new models or features

## Data Flow

```
Raw CSV/Generated Dataset
        ↓
   Data Loader (data_loader.py)
   - Load from file/Kaggle
   - Handle missing values
   - Normalize/Scale
   - SMOTE balancing
        ↓
   Preprocessed Data (X, y, scaler, encoders)
        ↓
   Model Trainer (model_trainer.py)
   - Split into train/test
   - Train multiple models
   - Evaluate metrics
        ↓
   Trained Models + Performance Metrics
        ↓
   Explainability (explainability.py)
   - SHAP explanation
   - LIME explanation
   - Feature importance
        ↓
   Streamlit Dashboard (app.py)
   - Display results
   - Interactive UI
   - Real-time predictions
```

## Key Modules

### 1. data_loader.py
**Responsibilities**:
- Load UPI dataset generation
- Kaggle dataset integration
- Data preprocessing pipeline
- Transaction validation

### 2. model_trainer.py
**Responsibilities**:
- Train ML models
- Evaluate performance
- Feature importance extraction
- Fraud prediction

### 3. explainability.py
**Responsibilities**:
- SHAP explanations
- LIME explanations
- Visualization generation
- Anomaly detection

### 4. digital_twin.py
**Responsibilities**:
- Simulate fraud scenarios
- Normal transaction patterns
- Attack simulations
- Risk analysis

### 5. utils.py
**Responsibilities**:
- Helper functions
- Model persistence
- Validation utilities
- Metric formatting

## Extension Points

### Adding a New Model

1. **Add to model_trainer.py**
2. **Update train_all_models()**
3. **Add to config.py**
4. **Update app.py**

### Adding a New Feature

1. Add to config.py
2. Include in preprocess_data()
3. Update dataset generation
4. Retrain models

## Optimization Tips

- Use Streamlit caching decorators
- Profile with cProfile
- Reduce sample size for testing
- Use faster models first

## Support

Check code docstrings and README.md for detailed information.

---

Happy coding!
