import pandas as pd
import sys
sys.path.insert(0, '.')

# Test the exact preprocessing flow
df = pd.read_csv('data/cicids.csv', nrows=10000)
print(f'Original columns (first 5): {df.columns[:5].tolist()}')

# Strip columns
df.columns = df.columns.str.strip()
print(f'After strip (first 5): {df.columns[:5].tolist()}')

# Check for label columns
possible_label_cols = [c for c in df.columns if c.lower() in ("label", "attack", "class", "isattack", "is_malicious")]
print(f'Possible label cols: {possible_label_cols}')

# Try fraud column creation
if possible_label_cols and 'fraud' not in df.columns:
    lab = possible_label_cols[0]
    print(f'Using label column: {lab}')
    print(f'Label dtype: {df[lab].dtype}')
    print(f'Sample label values: {df[lab].head().tolist()}')
    try:
        if df[lab].dtype == object:
            print('Creating fraud column from string labels...')
            df['fraud'] = df[lab].astype(str).str.upper().apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
        else:
            df['fraud'] = (df[lab] != 0).astype(int)
        print(f'Success! Fraud column created')
        print(f'Fraud distribution: {df["fraud"].value_counts().to_dict()}')
    except Exception as e:
        print(f'Error creating fraud column: {e}')
        import traceback
        traceback.print_exc()

# Check if fraud column exists
if 'fraud' in df.columns:
    print('Fraud column found in dataframe')
else:
    print('ERROR: Fraud column NOT found!')
