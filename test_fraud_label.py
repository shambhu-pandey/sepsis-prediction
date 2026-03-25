import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from config import DATA_CONFIG

# Load CICIDS raw
df = pd.read_csv('data/cicids.csv', nrows=10000)
df.columns = df.columns.str.strip()
print(f'Raw label distribution: {df["Label"].value_counts().to_dict()}')

# Manually do what preprocess_data does
df_test = df.copy()

# Create fraud column
lab = 'Label'
print(f'\nCreating fraud column from "{lab}":')
print(f'Sample values before conversion: {df_test[lab].head(10).tolist()}')

df_test['fraud'] = df_test[lab].astype(str).str.upper().apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
print(f'After conversion: {df_test["fraud"].value_counts().to_dict()}')

# Check if the issue is in the upper/strip
print(f'\nDebug sample:')
for i in range(5):
    raw = df_test.loc[i, lab]
    upper = str(raw).upper()
    stripped = upper.strip()
    result = 0 if stripped == 'BENIGN' else 1
    print(f'  {i}: "{raw}" -> upper: "{upper}" -> strip: "{stripped}" -> result: {result}')

# Now test the full preprocessing
print('\n\n=== Full Preprocessing Flow ===')
from modules.data_loader import load_dataset, preprocess_data

df2 = load_dataset("cicids", file_path="data/cicids.csv", sample_size=10000)
print(f'1. After load_dataset: shape={df2.shape}')
df2.columns = df2.columns.str.strip()
print(f'2. After column strip, Label distribution: {df2["Label"].value_counts().to_dict()}')

X, y, scaler, label_encoders, feature_names, stats = preprocess_data(
    df2, handle_missing=True, normalize=True, balance=True, sample_size=None
)
print(f'3. After preprocess_data: X={X.shape}, y distribution: {y.value_counts().to_dict()}')
print(f'   Stats: {stats}')
