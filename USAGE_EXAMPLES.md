# Usage Examples & Scenarios

Complete examples showing how to use the Fraud Detection Dashboard in various scenarios.

---

## Quick Start Examples

### Example 1: Running the Dashboard

#### Scenario: First-time user wants to explore fraud detection

```bash
# Step 1: Navigate to project directory
cd FRAUD_DETECTION

# Step 2: Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Step 3: Run the dashboard
streamlit run app.py

# Step 4: Open browser to http://localhost:8501
```

**Expected Flow:**
1. Dashboard loads with empty state
2. Navigate to Section 1 (Data Handling)
3. Click "Load & Preprocess Data" (defaults: 5000 records, 2% fraud)
4. Dataset is generated and statistics displayed
5. Navigate to Section 2 (Model Training)
6. Click "Train All Models" (takes 2-3 minutes)
7. View model comparison metrics
8. Navigate to Section 3 (Fraud Detection)
9. Try manual entry or sample transaction
10. Get fraud prediction with risk score

---

## Programmatic Usage Examples

### Example 2: Using Modules Independently

#### Scenario: Data scientist wants to train and evaluate models

```python
from modules.data_loader import generate_upi_dataset, preprocess_data, split_data
from modules.model_trainer import train_random_forest, evaluate_model
import pandas as pd

# Step 1: Generate dataset
print("Generating dataset...")
df = generate_upi_dataset(num_records=10000, fraud_rate=0.03)

# Step 2: Preprocess
print("Preprocessing...")
X, y, scaler, encoders, features = preprocess_data(
    df,
    handle_missing=True,
    normalize=True,
    balance=True
)

# Step 3: Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Step 4: Train specific model
print("Training Random Forest...")
model = train_random_forest(X_train, y_train)

# Step 5: Evaluate
metrics, predictions, probabilities = evaluate_model(
    model,
    X_test,
    y_test
)

# Step 6: Display results
print("Results:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### Example 3: Batch Fraud Detection

#### Scenario: Process multiple transactions from a CSV file

```python
import pandas as pd
from modules.data_loader import validate_and_prepare_transaction
from modules.model_trainer import predict_fraud
from modules.utils import load_latest_model, generate_risk_score

# Load model
model = load_latest_model("xgboost")
if model is None:
    print("No trained models found!")
    exit()

# Load transactions
transactions_df = pd.read_csv("transactions_to_check.csv")

results = []

# Process each transaction
for idx, row in transactions_df.iterrows():
    transaction = row.to_dict()
    
    # Validate and prepare
    is_valid, X_prepared, orig_data = validate_and_prepare_transaction(
        transaction,
        scaler=None,  # Set if you have saved scaler
        label_encoders={},
        feature_names=None
    )
    
    if is_valid:
        # Make prediction
        prediction, probability = predict_fraud(model, X_prepared)
        risk_info = generate_risk_score(probability)
        
        results.append({
            "transaction_id": transaction.get("transaction_id", ""),
            "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "probability": probability,
            "risk_score": risk_info["risk_score"],
            "risk_level": risk_info["risk_level"]
        })
    else:
        print(f"Skipped invalid transaction {idx}: {X_prepared}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("fraud_predictions.csv", index=False)
print(f"Processed {len(results)} transactions")
```

### Example 4: Model Comparison

#### Scenario: Compare all trained models with detailed metrics

```python
from modules.model_trainer import train_all_models, evaluate_model
from modules.data_loader import generate_upi_dataset, preprocess_data, split_data
import pandas as pd
import matplotlib.pyplot as plt

# Prepare data
df = generate_upi_dataset(5000)
X, y, _, _, _ = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train all models
results, models = train_all_models(X_train, y_train, X_test, y_test)

# Create comparison dataframe
comparison_df = pd.DataFrame(results).T
print(comparison_df.round(4))

# Visualize
import plotly.express as px

fig = px.bar(
    comparison_df.reset_index(),
    x="index",
    y=["Accuracy", "Precision", "Recall", "F1-Score"],
    title="Model Comparison",
    barmode="group"
)
fig.show()

# Find best model
best_model = max(results.items(), key=lambda x: x[1]["ROC-AUC"])
print(f"Best model: {best_model[0]} with ROC-AUC: {best_model[1]['ROC-AUC']:.4f}")
```

### Example 5: Feature Importance Analysis

#### Scenario: Identify which features are most important for fraud detection

```python
from modules.model_trainer import get_feature_importance
from modules.data_loader import generate_upi_dataset, preprocess_data, split_data
from modules.model_trainer import train_random_forest
import matplotlib.pyplot as plt

# Prepare and train
df = generate_upi_dataset(5000)
X, y, _, _, features = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_random_forest(X_train, y_train)

# Get feature importance
importance_dict = get_feature_importance(model, features, "Random Forest")

# Display top 10
print("Top 10 Important Features:")
for i, (feature, importance) in enumerate(list(importance_dict.items())[:10], 1):
    bar_length = int(importance * 50)
    bar = "█" * bar_length
    print(f"{i:2d}. {feature:20s} {bar} {importance:.4f}")

# Plot
plt.figure(figsize=(10, 6))
features_list = list(importance_dict.items())[:10]
plt.barh([f[0] for f in features_list], [f[1] for f in features_list])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance - Random Forest")
plt.tight_layout()
plt.show()
```

### Example 6: Digital Twin Simulation

#### Scenario: Simulate fraud attack scenarios and analyze risk

```python
from modules.digital_twin import DigitalTwinSimulator
import pandas as pd

# Initialize simulator
simulator = DigitalTwinSimulator()

# Example 1: Simulate normal transaction
print("=" * 60)
print("NORMAL TRANSACTION SCENARIO")
print("=" * 60)

normal_scenario = simulator.get_normal_scenario("Regular P2P Transfer")
normal_txn = simulator.simulate_transaction_flow(normal_scenario)
normal_risk = DigitalTwinSimulator.analyze_transaction_risk(normal_txn)

print(f"Scenario: {normal_scenario['name']}")
print(f"Description: {normal_scenario['description']}")
print(f"Transaction Amount: ₹{normal_txn['amount']:,.2f}")
print(f"Device Age: {normal_txn['device_age_days']:.0f} days")
print(f"Location Change: {'Yes' if normal_txn['location_change_indicator'] else 'No'}")
print(f"Risk Score: {normal_risk['risk_score']}/100")
print(f"Assessment: {normal_risk['summary']}")

# Example 2: Simulate fraud transaction
print("\n" + "=" * 60)
print("FRAUD SCENARIO")
print("=" * 60)

fraud_scenario = simulator.get_fraud_scenario("Compromised New Device")
fraud_txn = simulator.simulate_transaction_flow(fraud_scenario)
fraud_risk = DigitalTwinSimulator.analyze_transaction_risk(fraud_txn)

print(f"Scenario: {fraud_scenario['name']}")
print(f"Description: {fraud_scenario['description']}")
print(f"Transaction Amount: ₹{fraud_txn['amount']:,.2f}")
print(f"Device Age: {fraud_txn['device_age_days']:.0f} days")
print(f"Location Change: {'Yes' if fraud_txn['location_change_indicator'] else 'No'}")
print(f"Risk Score: {fraud_risk['risk_score']}/100")
print(f"Assessment: {fraud_risk['summary']}")

if fraud_risk['risk_factors']:
    print("Risk Factors:")
    for factor, detail in fraud_risk['risk_factors']:
        print(f"  • {factor}: {detail}")

# Example 3: Simulate attack scenario
print("\n" + "=" * 60)
print("ATTACK SIMULATION - CREDENTIAL COMPROMISE")
print("=" * 60)

attack_sequence = simulator.simulate_attack_scenario("credential_compromise")
attack_df = pd.DataFrame(attack_sequence)

print(attack_df.to_string())
```

### Example 7: SHAP Explanations

#### Scenario: Explain why a specific transaction was flagged as fraud

```python
from modules.data_loader import generate_upi_dataset, preprocess_data, split_data
from modules.model_trainer import train_random_forest
from modules.explainability import generate_shap_explanation
import pandas as pd

# Prepare data
df = generate_upi_dataset(5000)
X, y, _, _, features = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train model
model = train_random_forest(X_train, y_train)

# Select transaction to explain (first fraud case from test set)
fraud_indices = y_test[y_test == 1].index
if len(fraud_indices) > 0:
    idx = fraud_indices[0]
    X_explain = X_test.iloc[[idx]]
    
    # Generate SHAP explanation
    print("Generating SHAP explanation...")
    shap_result = generate_shap_explanation(
        model,
        X_train,
        X_explain,
        features,
        model_type="classification"
    )
    
    if shap_result:
        print("\n" + "=" * 60)
        print(f"SHAP Explanation for Transaction {idx}")
        print("=" * 60)
        print(f"Base Value: {shap_result.get('base_value', 'N/A'):.4f}")
        print("\nTop Features Contributing to FRAUD prediction:")
        
        shap_values = shap_result.get("shap_values", {})
        for i, (feature, contribution) in enumerate(list(shap_values.items())[:5], 1):
            direction = "↑ Increases" if contribution > 0 else "↓ Decreases"
            print(f"{i}. {feature:20s} {direction} fraud prob by {abs(contribution):.4f}")
```

### Example 8: Custom Configuration

#### Scenario: Run dashboard with custom parameters

```python
# Edit config.py to customize:

# Change data parameters
DATA_CONFIG = {
    "test_size": 0.3,        # 70% train, 30% test
    "random_state": 42,
    "smote_random_state": 42,
    "sample_size": 20000,   # Process up to 20k records
}

# Change model parameters
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,   # More trees for better accuracy
        "max_depth": 20,       # Deeper trees
        "random_state": 42,
        "n_jobs": -1
    },
    # ... other models
}

# Change UPI dataset parameters
UPI_CONFIG = {
    "num_records": 10000,     # Generate 10k transactions
    "fraud_rate": 0.05,       # 5% fraud rate
    "min_amount": 100,        # Min ₹100
    "max_amount": 100000,     # Max ₹100k
}

# Then restart: streamlit run app.py
```

---

## Real-World Scenarios

### Scenario 1: Financial Institution Daily Monitoring

```python
"""
Daily batch fraud detection for all transactions
"""
import pandas as pd
from datetime import datetime, timedelta

# Configuration
batch_size = 100000
risk_threshold = 0.7

# Load today's transactions
today = datetime.now()
transactions = load_transactions_from_database(
    start_date=today.date(),
    end_date=today.date()
)

# Process in batches
results = []
for i in range(0, len(transactions), batch_size):
    batch = transactions.iloc[i:i+batch_size]
    
    # Score batch
    batch_scores = score_transactions(batch)
    
    # Flag high-risk transactions
    high_risk = batch[batch_scores > risk_threshold]
    
    results.append(high_risk)

# Alert team
flagged_transactions = pd.concat(results)
send_alert_to_fraud_team(flagged_transactions)

# Log results
log_performance_metrics({
    "date": today,
    "transactions_processed": len(transactions),
    "transactions_flagged": len(flagged_transactions),
    "precision": calculate_precision(),
    "recall": calculate_recall()
})
```

### Scenario 2: Real-Time API Endpoint

```python
"""
Flask API for real-time fraud detection
"""
from flask import Flask, request, jsonify
from modules.model_trainer import predict_fraud
from modules.utils import load_latest_model, generate_risk_score

app = Flask(__name__)

# Load model at startup
model = load_latest_model("xgboost")

@app.route('/predict', methods=['POST'])
def predict_transaction():
    """Endpoint to predict fraud for a single transaction"""
    try:
        data = request.json
        
        # Prepare transaction
        is_prepared, X_transaction, _ = validate_and_prepare_transaction(
            data,
            scaler=None,
            label_encoders={},
            feature_names=None
        )
        
        if not is_prepared:
            return jsonify({"error": "Invalid transaction"}), 400
        
        # Make prediction
        prediction, probability = predict_fraud(model, X_transaction)
        risk_info = generate_risk_score(probability)
        
        return jsonify({
            "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "probability": probability,
            "risk_score": risk_info["risk_score"],
            "risk_level": risk_info["risk_level"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)
```

**Usage:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000,
    "sender": "USER001",
    "receiver": "USER999",
    "device_id": "DEV123",
    "time": "2026-02-28T15:30:45",
    "transaction_type": "P2P"
  }'
```

### Scenario 3: Weekly Model Retraining

```python
"""
Automated weekly model retraining with new data
"""
import schedule
import time
from datetime import datetime, timedelta

def retrain_models():
    """Retrain models with latest data"""
    print(f"[{datetime.now()}] Starting model retraining...")
    
    # Load latest data (last 2 weeks)
    df = load_transactions_from_db(
        days=14,
        include_fraud_label=True
    )
    
    # Preprocess
    X, y, scaler, encoders, features = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train
    results, models = train_all_models(X_train, y_train, X_test, y_test)
    
    # Log improvements
    log_retraining_results({
        "date": datetime.now(),
        "samples": len(X),
        "metrics": results
    })
    
    # Save best model
    best_model = max(results.items(), key=lambda x: x[1]["ROC-AUC"])
    save_model(models[best_model[0]]["model"], f"best_{best_model[0]}")
    
    print(f"[{datetime.now()}] Retraining complete. Best model: {best_model[0]}")

# Schedule weekly retraining every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(retrain_models)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Advanced Examples

### Example 9: Model Ensemble for Better Accuracy

```python
"""
Combine multiple models for improved fraud detection
"""
import numpy as np

class FraudDetectionEnsemble:
    def __init__(self, models_dict):
        """Initialize with trained models"""
        self.models = models_dict
        self.weights = self._calibrate_weights()
    
    def _calibrate_weights(self):
        """Calculate weights based on ROC-AUC"""
        # Weight by performance
        weights = {}
        total_auc = 0
        for name, model in self.models.items():
            # Get AUC from model metadata
            auc = model.get("auc", 0.8)
            weights[name] = auc
            total_auc += auc
        
        # Normalize
        return {k: v/total_auc for k, v in weights.items()}
    
    def predict(self, X):
        """Ensemble prediction"""
        predictions = []
        weights_list = []
        
        for model_name, model_data in self.models.items():
            model = model_data["model"]
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0, 1]
            else:
                proba = model.predict(X)[0]
            
            predictions.append(proba)
            weights_list.append(self.weights[model_name])
        
        # Weighted average
        ensemble_proba = np.average(predictions, weights=weights_list)
        
        return int(ensemble_proba > 0.5), ensemble_proba

# Usage
ensemble = FraudDetectionEnsemble(trained_models)
prediction, probability = ensemble.predict(X_test_sample)
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Using Different Scalers
```python
# WRONG: Different scaler on test data
X_train_scaled = scaler1.fit_transform(X_train)
X_test_scaled = scaler2.fit_transform(X_test)  # Wrong!

# CORRECT: Use same scaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform, not fit_transform
```

### Pitfall 2: Imbalanced Data Metrics
```python
# WRONG: Only using accuracy
accuracy = accuracy_score(y_true, y_pred)  # Misleading with imbalanced data

# CORRECT: Use multiple metrics
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)
```

### Pitfall 3: Data Leakage
```python
# WRONG: Fit scaler on all data before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leakage!
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Split first, then fit
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Support

For more examples and use cases, check:
- `quick_start.py` - Standalone demo script
- `README.md` - Documentation
- Module docstrings - Function-level examples
- `tests/` - Unit test examples (future)

---

**Happy coding!**
