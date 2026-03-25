import pandas as pd
import sys
import numpy as np

# Test load
print('Loading cicids.csv...')
df = pd.read_csv('data/cicids.csv', nrows=5000)

# Strip column names
df.columns = df.columns.str.strip()
print(f'Columns after strip: {df.columns.tolist()[:10]}')

# Create fraud column
print('\nCreating fraud column...')
label_col = 'Label'
print(f'Label unique values: {df[label_col].unique()}')
df['fraud'] = df[label_col].astype(str).str.upper().apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
print(f'Fraud distribution: {df["fraud"].value_counts().to_dict()}')

# Check which columns will be kept
sys.path.insert(0, '.')
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

y = df['fraud'].copy()
X = df.drop('fraud', axis=1)
print(f'\nX shape before feature selection: {X.shape}')

# Check what features are available vs what we want
available_numeric = [f for f in NUMERIC_FEATURES if f in X.columns]
available_categorical = [f for f in CATEGORICAL_FEATURES if f in X.columns]

print(f'Available numeric features: {len(available_numeric)}')
print(f'Available categorical features: {len(available_categorical)}')
print(f'Total features to use: {len(available_numeric) + len(available_categorical)}')
print(f'First 10 available numeric: {available_numeric[:10]}')
print(f'Available categorical: {available_categorical}')

# Check for non-numeric columns that will be dropped
print(f'\nAll X columns: {X.columns.tolist()}')
print(f'Dropping non-numeric/non-categorical columns...')
X_filtered = X[available_numeric + available_categorical]
print(f'X shape after feature selection: {X_filtered.shape}')
print(f'X dtypes:\n{X_filtered.dtypes}')

# Check for infinite values
print(f'\nChecking for infinite/NaN values...')
print(f'NaN count: {X_filtered.isna().sum().sum()}')
print(f'Infinite count: {np.isinf(X_filtered.select_dtypes(include=[np.number])).sum().sum()}')
