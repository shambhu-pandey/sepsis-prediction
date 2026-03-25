# Fraud Detection Dashboard

**Digital Twin–Enabled Framework for Forecasting and Mitigating Fraud with UPI Integration**

A comprehensive, production-ready Streamlit-based fraud detection dashboard that combines machine learning, digital twin simulations, and explainable AI to detect and prevent fraudulent UPI transactions.

---

## Features

### 1. **Intelligent Data Handling**
- Automatic UPI dataset generation with realistic transaction patterns
- Support for PaySim and CICIDS2017 datasets from Kaggle
- CSV file upload for custom datasets
- Comprehensive data preprocessing:
  - Missing value imputation
  - Feature normalization (StandardScaler)
  - Class imbalance handling (SMOTE)
  - Automatic train-test split

### 2. **Multi-Model Machine Learning**
- **Logistic Regression**: Fast baseline with interpretable coefficients
- **Random Forest**: Captures non-linear patterns
- **XGBoost**: Superior performance with gradient boosting
- **Autoencoder**: Anomaly detection using neural networks
- Model comparison with metrics:
# Fraud Detection Dashboard

**Digital Twin–Enabled Framework for Forecasting and Mitigating Fraud with UPI Integration**

A comprehensive, production-ready Streamlit-based fraud detection dashboard that combines machine learning, digital twin simulations, and explainable AI to detect and prevent fraudulent UPI transactions.

---

## Features

### 1. Intelligent Data Handling
- Automatic UPI dataset generation with realistic transaction patterns
- Support for PaySim and CICIDS2017 datasets from Kaggle
- CSV file upload for custom datasets
- Comprehensive data preprocessing:
  - Missing value imputation
  - Feature normalization (StandardScaler)
  - Class imbalance handling (SMOTE)
  - Automatic train-test split

### 2. Multi-Model Machine Learning
- Logistic Regression: Fast baseline with interpretable coefficients
- Random Forest: Captures non-linear patterns
- XGBoost: Superior performance with gradient boosting
- Autoencoder: Anomaly detection using neural networks
- Model comparison with metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

### 3. Real-Time Fraud Detection
- Interactive transaction input form
- One-click sample transaction generation
- Instant fraud probability prediction
- Risk score calculation (0-100)
- Fraud/Legitimate classification with confidence

### 4. Explainable AI (XAI)
- SHAP (SHapley Additive exPlanations): Feature contribution analysis
- LIME (Local Interpretable Model-agnostic Explanations): Local interpretability
- Feature importance rankings
- Anomaly Detection (unusual amounts, new devices, location anomalies, suspicious times)

### 5. Digital Twin Simulation
- Normal transaction scenario modeling
- Fraud attack pattern simulation (credential compromise, SIM swap, card testing)
- Risk factor analysis per transaction
- Transaction flow visualization

### 6. Professional UI/UX
- Clean, intuitive Streamlit interface
- Non-technical user-friendly design
- Interactive visualizations (Plotly)
- Real-time metric displays

---

## Project Structure

```
FRAUD_DETECTION/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Project dependencies
├── config.py                           # Configuration settings
├── modules/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading & preprocessing
│   ├── model_trainer.py               # ML model training & evaluation
│   ├── explainability.py              # SHAP & LIME integration
│   ├── digital_twin.py                # Digital twin simulation
│   └── utils.py                       # Utility functions
├── data/
│   └── [Generated UPI datasets]
├── models/
│   └── [Saved trained models]
└── README.md                          # This file
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone or download the project
   ```bash
   cd FRAUD_DETECTION
   ```

2. Create virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the dashboard
   ```bash
   streamlit run app.py
   ```

### Kaggle datasets (PaySim, CICIDS2017)

The app can automatically download PaySim and CICIDS2017 datasets from Kaggle when you select those options. To enable this you must install the `kaggle` package and provide your Kaggle API credentials.

1. Install the Kaggle package:

```bash
pip install kaggle
```

2. Place your `kaggle.json` (containing your API key) in `~/.kaggle/kaggle.json` or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. See https://www.kaggle.com/docs/api for details.

3. Run Streamlit as usual. The dashboard will attempt to download candidate dataset slugs (configured in `config.py`) and will use a sampled subset (default 10,000 rows) for training to keep runtime reasonable.

If automatic download fails (no credentials or dataset not found), the app falls back to using a local CSV if provided, or the synthetic UPI generator.

Access the dashboard: Open your browser and navigate to `http://localhost:8501`

---

## Usage Guide

### Section 1: Dataset Handling & Preprocessing

1. Choose data source:
   - Synthetic UPI – generates the built‑in fake UPI dataset (default, quick demo).
   - PaySim (Kaggle/local) – loads the PaySim fraud dataset via Kaggle or a local CSV.
   - Upload UPI CSV – supply your own UPI‑formatted file.
   - CICIDS2017 CSV – supply a CICIDS‑style intrusion dataset with a `Label` or `fraud` column.

2. Configure dataset:
   - For synthetic data: set the number of records (1,000 – 10,000) and fraud rate (0.5% – 10%).
   - For external CSVs: specify a maximum number of rows to load (e.g. 10,000) to keep training fast.
   - You may either upload a file or type/paste a local path; if both are provided the upload takes precedence.

3. Preprocessing options:
   - Handle missing values
   - Normalize features
   - Balance dataset (SMOTE)

4. Click "Load & Preprocess Data"
  * The synthetic generator creates a `fraud_reason` column explaining why a row is tagged as fraud (e.g. high amount, new device). When using external datasets such as PaySim or CICIDS2017 the column may not exist.
  * SMOTE balancing may expand the dataset size; the dashboard shows both original and post‑balance counts.

### Section 2: Machine Learning Models

1. Click "Train All Models" to train:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Autoencoder

2. Review performance metrics:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Visual comparisons (bar charts, graphs)
   - Feature importance analysis

3. Select best model for predictions

### Section 3: Fraud Detection Demo

1. Enter transaction details OR select "Sample Transaction"

2. Choose prediction model

3. Click "Predict Fraud"

4. Get results:
   - Prediction: Fraud or Legitimate
   - Confidence score
   - Risk score (0-100)
   - Detected anomalies
   - Feature contribution (SHAP)

### Section 4: Digital Twin Simulation

Tab 1: Scenario Comparison
- View normal transaction patterns
- View common fraud attack patterns

Tab 2: Attack Simulation
- Simulate credential compromise
- Simulate SIM swap attacks
- Simulate card testing attacks

Tab 3: Risk Analysis
- Analyze transaction risk factors
- Get detailed risk breakdowns

### Section 5: Dashboard Insights

- Key statistics and metrics
- Best practices for fraud detection
- System information

---

## Dataset Format

### UPI Transaction Fields
```
{
    "transaction_id": "TXN0000001",          # Unique identifier
    "sender": "USER001",                     # Sender UPI ID
    "receiver": "USER999",                   # Receiver UPI ID
    "amount": 5000.50,                         # Transaction amount (₹)
    "device_id": "DEV123456",                # Device identifier
    "sender_device_type": "Android",         # Device OS
    "receiver_device_type": "iOS",           # Receiver device
    "location": "Location_42",               # Geographic location
    "time": "2026-02-28T15:30:45",           # Transaction timestamp
    "transaction_type": "P2P",               # "P2P", "Merchant", "Bill Payment", etc.
    "transaction_count_24h": 5,                # Transactions in last 24 hours
    "location_change_indicator": 0,            # 1 if location changed, 0 otherwise
    "device_age_days": 365,                    # Days since device registration
```
- **Best for**: Balancing speed and accuracy

### 3. XGBoost
- **Algorithm**: Gradient boosting
- **Pros**: Best performance, handles imbalanced data well
- **Cons**: Requires tuning, slower training
- **Best for**: Maximum accuracy requirements

### 4. Autoencoder
- **Algorithm**: Neural network for anomaly detection
- **Pros**: Detects novel fraud patterns, learns normal behavior
- **Cons**: Requires more data, hyperparameter tuning
- **Best for**: Capturing emerging fraud types

---

## Performance Metrics Explanation

### Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Meaning**: Percentage of correct predictions
- **Limitation**: Misleading with imbalanced data

### Precision
- **Formula**: TP / (TP + FP)
- **Meaning**: Of flagged frauds, how many are actually fraud?
- **Important for**: Minimizing false positives

### Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Meaning**: Of actual frauds, how many did we catch?
- **Important for**: Catching all fraud cases

### F1-Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Meaning**: Harmonic mean of precision and recall
- **Good for**: Overall model evaluation

### ROC-AUC
- **Meaning**: Area under ROC curve
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Good for**: Threshold-independent evaluation

---

## Explainability Features

### SHAP (SHapley Additive exPlanations)
- Show which features pushed prediction toward fraud/legitimate
- Feature contributions are additive
- Based on game theory principles
- More accurate but computationally expensive

### LIME (Local Interpretable Model-agnostic Explanations)
- Explain individual predictions
- Create local linear approximations
- Model-agnostic (works with any model)
- Faster than SHAP

### Anomaly Scoring
A utomatically detects:
- Unusually high transaction amounts
- Transactions from new devices
- Rapid location changes
- Suspicious transaction times (2-5 AM)
- Unusual transaction frequency

---

## Configuration Settings

Edit `config.py` to customize:

```python
# Data Configuration
DATA_CONFIG = {
    "test_size": 0.2,                    # Train-test split ratio
    "random_state": 42,                  # Reproducibility
    "smote_random_state": 42,            # SMOTE seed
    "sample_size": 50000,                # Max records to process
}

# Model Configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,             # Number of trees
        "max_depth": 15,                 # Tree depth limit
    },
    # ... other model configs
}

# UPI Dataset Configuration
UPI_CONFIG = {
    "num_records": 5000,                 # Records to generate
    "fraud_rate": 0.02,                  # 2% fraud rate
    "min_amount": 10,                    # Min transaction amount
    "max_amount": 50000,                 # Max transaction amount
}
```

---

## Security & Best Practices

### Data Handling
- No personal data stored permanently
- Synthetic dataset generation
- Feature normalization (hides raw values)
- Local processing only

### Model Deployment
- Models saved locally with timestamps
- Version control for reproducibility
- No external API calls for predictions
- Explainability for audit trails

### Fraud Detection Best Practices
1. **Understand false positive cost**: Declined legitimate transactions
2. **Understand false negative cost**: Undetected fraud
3. **Adjust threshold**: Based on business requirements
4. **Regular model updates**: Adapt to evolving fraud patterns
5. **Multi-layer defense**: Combine models for robustness
6. **Monitor performance**: Track metrics over time

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | ML algorithms |
| `xgboost` | Gradient boosting |
| `tensorflow` | Deep learning (autoencoder) |
| `imbalanced-learn` | SMOTE for class imbalance |
| `shap` | SHAP explanations |
| `lime` | LIME explanations |
| `matplotlib`, `seaborn` | Data visualization |
| `plotly` | Interactive charts |
| `joblib` | Model persistence |

---

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure all dependencies installed: `pip install -r requirements.txt`

### Issue: Models training very slowly
**Solution**: 
- Reduce sample size in Section 1
- Use faster model (Logistic Regression) first
- Run on GPU if available (for Autoencoder)

### Issue: SHAP/LIME explanations not showing
**Solution**: Try with smaller dataset or specific model

### Issue: Out of memory error
**Solution**: 
- Reduce `sample_size` in config.py
- Use fewer records in dataset generation
- Close other applications

### Issue: Port 8501 already in use
**Solution**: 
```bash
streamlit run app.py --logger.level=debug --server.port=8502
```

---

## Advanced Usage

### Using Custom Dataset

```python
import pandas as pd
from modules.data_loader import preprocess_data, split_data

# Load your data
df = pd.read_csv('your_data.csv')

# Preprocess
X, y, scaler, encoders, features = preprocess_data(df)

# Train
from modules.model_trainer import train_random_forest
model = train_random_forest(X_train, y_train)
```

### Loading Saved Models

```python
from modules.utils import load_latest_model

model = load_latest_model("random_forest")
if model:
    prediction = model.predict(new_data)
```

### Batch Prediction

```python
from modules.model_trainer import predict_fraud

predictions = []
for idx, row in df.iterrows():
    pred, prob = predict_fraud(model, row.reshape(1, -1))
    predictions.append({"prediction": pred, "probability": prob})
```

---

## Sample Output

### Model Comparison
```
Model                 Accuracy  Precision  Recall  F1-Score  ROC-AUC
─────────────────────────────────────────────────────────────────────
Logistic Regression    0.9742    0.9521    0.8934   0.9217   0.9865
Random Forest          0.9856    0.9723    0.9145   0.9425   0.9924
XGBoost                0.9901    0.9834    0.9312   0.9567   0.9956
Autoencoder            0.9823    0.9634    0.9089   0.9355   0.9905
```

### Fraud Prediction Example
```
Transaction Prediction: FRAUD
Probability: 94.23%
Risk Score: 94/100
Risk Level: VERY HIGH

Detected Anomalies:
• Unusually high amount: ₹45,000.00
• New device (age: 2 days)
• Location change detected
• Unusual transaction time: 3:00

Feature Contributions:
1. amount: +0.45 (high amounts increase fraud probability)
2. device_age_days: +0.32 (new devices suspicious)
3. location_change: +0.28 (location changes increase risk)
```

---

## Performance Optimization

### For Real-Time Predictions
```python
# Use fast model
use_model = "Logistic Regression"
```

### For Maximum Accuracy
```python
# Use high-performance model
use_model = "XGBoost"
```

### For Anomaly Detection
```python
# Use autoencoder
use_model = "Autoencoder"
```

---

## Support & Contribution

### Known Limitations
- LIME explanations slower than SHAP
- Autoencoder requires GPU for large datasets
- Kaggle dataset download requires API credentials
- Maximum ~50,000 records recommended for smooth operation

### Future Enhancements
- [ ] Real-time model retraining
- [ ] Multi-language support
- [ ] Database integration
- [ ] API endpoint deployment
- [ ] Mobile app version
- [ ] Deep reinforcement learning model
- [ ] Graph neural networks for transaction networks

---

## License

This project is provided as-is for educational and research purposes.

---

## Author Notes

This framework demonstrates:
1. **End-to-end ML pipeline**: Data → Models → Predictions → Explanations
2. **Professional deployment**: Production-ready Streamlit app
3. **Explainable AI**: SHAP/LIME for model transparency
4. **Digital twins**: Simulating fraud patterns and attack sequences
5. **Best practices**: Documentation, modularity, error handling

For questions or improvements, refer to code comments and docstrings.

---

## Educational Resources

### Understanding Fraud Detection
- Kaggle PaySim dataset paper
- CICIDS2017 network intrusion dataset
- SHAP documentation
- LIME explanations

### Related Topics
- Digital payments security
- UPI transaction flows
- Machine learning in fintech
- Explainable AI (XAI)

---

**Last Updated**: February 28, 2026  
**Version**: 1.0.0  
**Status**: Production Ready

---

### Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8502

# Run in headless mode (CI/CD)
streamlit run app.py --logger.level=error --client.showErrorDetails=false

# Access dashboard
http://localhost:8501
```

Enjoy the dashboard!
