import pandas as pd
import sys
sys.path.insert(0, '.')

from config import DATA_CONFIG
from modules.data_loader import load_dataset, preprocess_data, split_data
from modules.model_trainer import train_all_models

# Simulate the exact training flow
print("=== CICIDS Training Flow ===")
df = load_dataset("cicids", file_path="data/cicids.csv", sample_size=DATA_CONFIG.get("sample_size", 10000))
print(f'1. Loaded: {df.shape[0]} rows')

df.columns = df.columns.str.strip()
if ' Label' in df.columns or 'Label' in df.columns:
    col = 'Label' if 'Label' in df.columns else ' Label'
    print(f'2. Label distribution: {df[col].value_counts().to_dict()}')

try:
    X, y, scaler, label_encoders, feature_names, stats = preprocess_data(
        df, handle_missing=True, normalize=True, balance=True, sample_size=DATA_CONFIG.get("sample_size", 10000)
    )
    print(f'3. Preprocessed: X={X.shape}, y={y.shape}')
    print(f'   Fraud distribution: {y.value_counts().to_dict()}')
except Exception as e:
    print(f'3. ERROR in preprocessing: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f'4. Split: train={X_train.shape}, test={X_test.shape}')
    print(f'   Train fraud dist: {y_train.value_counts().to_dict()}')
    print(f'   Test fraud dist: {y_test.value_counts().to_dict()}')
except Exception as e:
    print(f'4. ERROR in split: {e}')
    sys.exit(1)

try:
    print('5. Training models...')
    results, models = train_all_models(X_train, y_train, X_test, y_test)
    print('6. Training completed successfully!')
    print('Results:')
    for m, metrics in results.items():
        print(f' - {m}: ROC-AUC={metrics.get("ROC-AUC"):.4f}, F1={metrics.get("F1-Score"):.4f}')
except Exception as e:
    print(f'5. ERROR in training: {e}')
    import traceback
    traceback.print_exc()
