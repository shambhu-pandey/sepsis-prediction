import pandas as pd
import numpy as np

# Load and test fraud column creation directly
df = pd.read_csv('data/cicids.csv', nrows=10000)
df.columns = df.columns.str.strip()

# Simulate random sampling like load_dataset does
df = df.sample(n=10000, random_state=42)
print(f'After shuffle: {df["Label"].value_counts().to_dict()}')

# Test the new fraud column conversion logic
df_temp = df["Label"].astype(str)
fraud = df_temp.str.strip().str.upper().eq('BENIGN').astype(int)
fraud = (1 - fraud)  # Invert

print(f'Fraud conversion result: {fraud.value_counts().to_dict()}')
print(f'Sample conversions:')
for i in range(10):
    orig = df.iloc[i]["Label"]
    f = fraud.iloc[i]
    print(f'  "{orig}" -> {f}')

# Now check if there are weird characters
print(f'\n\nLabel raw bytes (first 5):')
for i in range(5):
    label = df.iloc[i]["Label"]
    print(f'  {i}: repr={repr(label)}, bytes={label.encode("utf-8")}')
