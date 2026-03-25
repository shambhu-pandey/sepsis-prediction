"""
Quick Start Script - Demonstrates all dashboard features without UI
Useful for testing and debugging
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("FRAUD DETECTION DASHBOARD - QUICK START DEMO")
print("=" * 80)
print()

# Import modules
print("Loading modules...")
try:
    from modules.data_loader import generate_upi_dataset, preprocess_data, split_data
    from modules.model_trainer import train_all_models, get_feature_importance
    from modules.explainability import explain_prediction
    from modules.digital_twin import DigitalTwinSimulator
    from modules.utils import generate_risk_score, generate_sample_transaction
    print("Modules loaded successfully!\n")
except ImportError as e:
    print(f"Error loading modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Step 1: Obtain Dataset
print("Step 1 - Obtaining Dataset")
print("-" * 80)

# allow user to choose between synthetic and external datasets
choice = input(
    "Enter dataset type [synthetic/paysim/upi/cicids] (default synthetic): "
).strip().lower() or "synthetic"
file_path = None
if choice in ["paysim", "upi", "cicids"]:
    file_path = input("If you have a local CSV path, enter it now (or leave blank to skip): ").strip() or None

if choice == "synthetic":
    num = 5000
    rate = 0.02
    print(f"Generating {num} UPI transactions with {rate*100:.1f}% fraud rate...")
    df = generate_upi_dataset(num_records=num, fraud_rate=rate)
elif choice == "paysim":
    from modules.data_loader import load_dataset
    print("Attempting to load PaySim dataset (Kaggle or provided file)...")
    df = load_dataset("paysim", file_path=file_path, sample_size=10000)
elif choice == "upi":
    from modules.data_loader import load_dataset
    print("Loading UPI dataset from CSV...")
    df = load_dataset("upi", file_path=file_path, sample_size=10000)
elif choice == "cicids":
    from modules.data_loader import load_dataset
    print("Loading CICIDS2017 dataset from CSV...")
    df = load_dataset("cicids", file_path=file_path, sample_size=10000)
else:
    print(f"Unknown dataset choice '{choice}', falling back to synthetic.")
    df = generate_upi_dataset(num_records=5000, fraud_rate=0.02)

print(f"Dataset obtained! {len(df)} records loaded")
if 'fraud' in df.columns:
    orig_count = (df['fraud'] == 1).sum()
    legit_count = (df['fraud'] == 0).sum()
    print(f"   Fraud cases: {orig_count}  Legitimate: {legit_count}")
    print(f"   Fraud rate: {orig_count/len(df)*100:.2f}%")
print(f"   Features: {df.shape[1]}")
if 'fraud_reason' in df.columns:
    sample_reasons = df.loc[df['fraud'] == 1, 'fraud_reason'].value_counts().head(3)
    print("   Sample fraud reasons:")
    for reason, cnt in sample_reasons.items():
        print(f"      - {reason}: {cnt} cases")
print()
# Step 2: Preprocess Data
print("Step 2 - Preprocessing Data")
print("-" * 80)

print("Preprocessing data (handling missing values, normalization, SMOTE balancing)...")
X, y, scaler, label_encoders, feature_names, stats = preprocess_data(
    df,
    handle_missing=True,
    normalize=True,
    balance=True,
    sample_size=None
)

# explain class balancing
orig_total = stats.get("original_total", len(df))
orig_fraud = stats.get("original_fraud", 0)
print(f"   (original dataset: {orig_total} records, {orig_fraud} fraud cases)")
print("   Note: SMOTE balancing may change the total number of samples;"
      " fraudulent cases shown below correspond to the processed set.")

print(f"Data preprocessed!")
print(f"   Processed records: {len(X)}")
print(f"   Features: {len(feature_names)}")
print(f"   Feature list: {feature_names[:5]} ...")
print()

# Step 3: Split Data
print("Step 3 - Splitting Dataset")
print("-" * 80)

X_train, X_test, y_train, y_test = split_data(X, y)

print(f"Data split complete!")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"   Train fraud rate: {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
print(f"   Test fraud rate: {(y_test == 1).sum() / len(y_test) * 100:.2f}%")
print()

# Step 4: Train Models
print("Step 4 - Training Machine Learning Models")
print("-" * 80)

print("Training models (this may take 2-3 minutes)...")
print("   • Logistic Regression")
print("   • Random Forest")
print("   • XGBoost")
print("   • Autoencoder")

results, models = train_all_models(X_train, y_train, X_test, y_test)

print(f"\nAll models trained!")
print("\nModel Performance Comparison:")
print("-" * 80)

results_df = pd.DataFrame(results).T.round(4)
print(results_df.to_string())

print("\n" + "=" * 80)
best_model = max(results.items(), key=lambda x: x[1].get("ROC-AUC", 0))
print(f"Best Model: {best_model[0]} (ROC-AUC: {best_model[1].get('ROC-AUC', 0):.4f})")
print("=" * 80)
print()

# Step 5: Feature Importance
print("Step 5 - Analyzing Feature Importance")
print("-" * 80)

for model_name, model_data in list(models.items())[:2]:  # Show first 2 models
    try:
        importance = get_feature_importance(model_data["model"], feature_names, model_name)
        if importance:
            print(f"\n{model_name} - Top 5 Features:")
            for i, (feat, imp) in enumerate(list(importance.items())[:5], 1):
                print(f"   {i}. {feat}: {imp:.4f}")
    except Exception as e:
        print(f"   Could not compute: {e}")

print()

# Step 6: Digital Twin Simulation
print("Step 6 - Digital Twin Simulation")
print("-" * 80)

simulator = DigitalTwinSimulator()

print("\nNormal Scenario Example:")
normal_scenario = simulator.get_normal_scenario()
normal_txn = simulator.simulate_transaction_flow(normal_scenario)
normal_risk = DigitalTwinSimulator.analyze_transaction_risk(normal_txn)

print(f"   Scenario: {normal_scenario['name']}")
print(f"   Description: {normal_scenario['description']}")
print(f"   Amount: ₹{normal_txn['amount']:,.2f}")
print(f"   Risk Score: {normal_risk['risk_score']}/100")
print(f"   Assessment: {normal_risk['summary']}")

print("\nFraud Scenario Example:")
fraud_scenario = simulator.get_fraud_scenario()
fraud_txn = simulator.simulate_transaction_flow(fraud_scenario)
fraud_risk = DigitalTwinSimulator.analyze_transaction_risk(fraud_txn)

print(f"   Scenario: {fraud_scenario['name']}")
print(f"   Description: {fraud_scenario['description']}")
print(f"   Amount: ₹{fraud_txn['amount']:,.2f}")
print(f"   Risk Score: {fraud_risk['risk_score']}/100")
print(f"   Assessment: {fraud_risk['summary']}")

if fraud_risk['risk_factors']:
    print("   Risk Factors:")
    for factor, detail in fraud_risk['risk_factors']:
        print(f"      • {factor}: {detail}")

print()

# Step 7: Make Prediction
print("Step 7 - Making Fraud Prediction")
print("-" * 80)

print("\nGenerating sample transaction...")
sample_txn = generate_sample_transaction()

print(f"Transaction Details:")
print(f"   Amount: ₹{sample_txn['amount']:,.2f}")
print(f"   Type: {sample_txn['transaction_type']}")
print(f"   Sender: {sample_txn['sender']}")
print(f"   Receiver: {sample_txn['receiver']}")
print(f"   Device Age: {sample_txn['device_age_days']:.0f} days")
print(f"   Location Change: {'Yes' if sample_txn['location_change_indicator'] else 'No'}")
print(f"   Hour: {sample_txn['time'].hour}:00")

# Prepare for prediction
from modules.data_loader import validate_and_prepare_transaction

is_prepared, X_pred, _ = validate_and_prepare_transaction(
    sample_txn,
    scaler,
    label_encoders,
    feature_names
)

if is_prepared:
    # Use best model
    best_model_name = best_model[0]
    best_model_obj = models[best_model_name]["model"]
    model_type = "autoencoder" if best_model_name == "Autoencoder" else "classification"
    
    from modules.model_trainer import predict_fraud
    prediction, probability = predict_fraud(best_model_obj, X_pred, model_type)
    
    risk_info = generate_risk_score(probability)
    
    print(f"\nPrediction Results (Using {best_model_name}):")
    print(f"   Prediction: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")
    print(f"   Confidence: {probability:.2%}")
    print(f"   Risk Score: {risk_info['risk_score']}/100")
    print(f"   Risk Level: {risk_info['risk_level']}")
    print(f"   Description: {risk_info['description']}")

print()

# Step 8: Summary Statistics
print("Step 8 - Summary Statistics")
print("-" * 80)

print(f"\nSystem Status:")
print(f"   Models Trained: {len(models)}")
print(f"   Model Names: {', '.join(models.keys())}")
print(f"   Features Used: {len(feature_names)}")
print(f"   Dataset Samples: {len(X)}")
print(f"   Average Accuracy: {np.mean([m.get('Accuracy', 0) for m in results.values()]):.4f}")
print(f"   Average ROC-AUC: {np.mean([m.get('ROC-AUC', 0) for m in results.values()]):.4f}")

print()
print("=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print()
print("Next Steps:")
print("   1. Run: streamlit run app.py")
print("   2. Explore the interactive dashboard")
print("   3. Try different models and datasets")
print("   4. Make custom fraud predictions")
print("   5. Review explainability features")
print()
print("Documentation:")
print("   • README.md - Full documentation")
print("   • INSTALLATION.md - Setup instructions")
print("   • config.py - Configuration options")
print()
print("=" * 80)
