import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

from modules.model_trainer import build_model_pipeline
from sklearn.linear_model import LogisticRegression

def test_pipeline_imbalanced():
    print("Testing imbalanced dataset (PaySim style)...")
    df = pd.DataFrame({
        "amount": np.random.rand(100),
        "oldbalanceOrg": np.random.rand(100),
        "type": ["TRANSFER"] * 100
    })
    y = pd.Series([0] * 95 + [1] * 5)
    
    # This should include SMOTETomek
    pipe = build_model_pipeline("Logistic Regression", LogisticRegression(), df, y, ds_key="paysim")
    
    step_names = [s[0] for s in pipe.steps]
    print(f"Pipeline steps: {step_names}")
    assert "smote_tomek" in step_names
    print("✓ SMOTETomek included for imbalanced dataset.")

def test_pipeline_balanced():
    print("\nTesting balanced dataset (CICIDS style)...")
    df = pd.DataFrame({
        "amount": np.random.rand(100),
        "oldbalanceOrg": np.random.rand(100),
        "type": ["TRANSFER"] * 100
    })
    y = pd.Series([0] * 50 + [1] * 50)
    
    # This should NOT include SMOTETomek because ds_key="cicids"
    pipe = build_model_pipeline("Logistic Regression", LogisticRegression(), df, y, ds_key="cicids")
    
    step_names = [s[0] for s in pipe.steps]
    print(f"Pipeline steps: {step_names}")
    assert "smote_tomek" not in step_names
    print("✓ SMOTETomek skipped for balanced dataset.")

def test_leakage_assertion():
    print("\nTesting leakage assertion...")
    df = pd.DataFrame({
        "amount": np.random.rand(100),
        "fraud": [0] * 100 # LEAKAGE!
    })
    y = pd.Series([0] * 100)
    
    try:
        build_model_pipeline("Logistic Regression", LogisticRegression(), df, y, ds_key="paysim")
        print("✗ Leakage assertion failed (did not raise Exception)")
    except AssertionError as e:
        print(f"✓ Leakage assertion caught: {e}")

if __name__ == "__main__":
    try:
        test_pipeline_imbalanced()
        test_pipeline_balanced()
        test_leakage_assertion()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
