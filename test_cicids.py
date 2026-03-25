import pandas as pd
import sys

# Test load
print('Loading cicids.csv...')
df = pd.read_csv('data/cicids.csv', nrows=5000)
print(f'Shape: {df.shape}')
print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')

# Check for label column
label_col = None
for col in df.columns:
    if 'label' in col.lower():
        label_col = col
        print(f'\nFound label column: "{col}"')
        print(f'Data type: {df[col].dtype}')
        print(f'Unique values: {df[col].unique()[:10]}')
        break

if label_col is None:
    print('\nNo label column found!')
    print('Available columns:', df.columns.tolist())

# Now try preprocessing
sys.path.insert(0, '.')
from modules.data_loader import preprocess_data

print('\n\nTrying preprocessing...')
try:
    X, y, scaler, label_encoders, feature_names, stats = preprocess_data(
        df, handle_missing=True, normalize=True, balance=False, sample_size=None
    )
    print(f'Success! X shape: {X.shape}, y shape: {y.shape}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
