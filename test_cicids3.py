import pandas as pd
import numpy as np
import sys

# Test load
print('Loading cicids.csv...')
df = pd.read_csv('data/cicids.csv', nrows=5000)

# Strip column names
df.columns = df.columns.str.strip()

# Create fraud column
label_col = 'Label'
df['fraud'] = df[label_col].astype(str).str.upper().apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)

y = df['fraud'].copy()
X = df.drop('fraud', axis=1)

# Auto-detect numeric columns
numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
X = X[numeric_cols]

print(f'\nX shape: {X.shape}')
print(f'Data types:\n{X.dtypes}')

# Check for infinite/NaN values
print(f'\nInfinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}')
print(f'NaN values: {X.isna().sum().sum()}')

# Show problem areas
if np.isinf(X.select_dtypes(include=[np.number])).sum().sum() > 0:
    print('\nColumns with infinite values:')
    for col in X.columns:
        inf_count = np.isinf(X[col]).sum()
        if inf_count > 0:
            print(f'  {col}: {inf_count}')

# Now try preprocessing
sys.path.insert(0, '.')
from modules.data_loader import preprocess_data

print('\n\nTrying preprocessing with updated function...')
try:
    X_proc, y_proc, scaler, label_encoders, feature_names, stats = preprocess_data(
        df, handle_missing=True, normalize=True, balance=False, sample_size=None
    )
    print(f'Success! X shape: {X_proc.shape}, y shape: {y_proc.shape}')
    print(f'Features: {feature_names[:10]}')
    print(f'Stats: {stats}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
