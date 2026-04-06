
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from modules.data_loader import _read_csv, _find_target, _binarize_target, extract_features
from config import DATASET_META, MODELS_DIR, RAW_FEATURES_BY_DATASET

def diagnose():
    results = []
    for key in DATASET_META.keys():
        meta = DATASET_META[key]
        path = meta["file"]
        if not os.path.exists(path):
            # Try alt
            for alt in meta.get("alt_files", []):
                if os.path.exists(alt):
                    path = alt
                    break
        
        if not os.path.exists(path):
            results.append(f"Skipping {key}: file {path} not found.")
            continue
            
        results.append(f"\n--- Testing Dataset: {key} ---")
        try:
            # 1. Load raw
            df = _read_csv(path)
            if df is None:
                results.append(f"Failed to read {path}")
                continue
            
            # 2. Extract
            X, y = extract_features(df)
            
            # Select only features specified in config (to match training)
            required = RAW_FEATURES_BY_DATASET.get(key)
            if required:
                X = X[[c for c in required if c in X.columns]]
            
            # 3. Numeric subset
            X_num = X.select_dtypes(include=[np.number])
            if X_num.empty:
                results.append(f"No numeric features for {key}")
                continue
            
            # 4. Correlation check
            temp = X_num.copy()
            temp['target'] = y.values
            corrs = temp.corr()['target'].abs().sort_values(ascending=False)
            
            results.append("Top absolute correlations with target:")
            results.append(str(corrs.head(10)))
            
            # 5. Perfect correlation search
            perfect = corrs[corrs > 0.999].index.tolist()
            if 'target' in perfect: perfect.remove('target')
            if perfect:
                results.append(f"🔥🔥🔥 PERFECT LEAKAGE DETECTED in columns: {perfect}")
            
        except Exception as e:
            results.append(f"Error: {str(e)}")

    with open("diag_output.txt", "w") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    diagnose()
