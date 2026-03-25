import pandas as pd
import sys
sys.path.insert(0, '.')

from config import DATA_CONFIG

# Load and sample exactly like the script does
df = pd.read_csv('data/cicids.csv')
df.columns = df.columns.str.strip()  
df = df.sample(n=10000, random_state=DATA_CONFIG['random_state'])
print(f'1. After shuffling: {df["Label"].value_counts().to_dict()}')

# Copy for preprocessing simulation
df2 = df.copy()
df2.columns = df2.columns.str.strip()  # Already stripped, but do it again

# Create fraud column as preprocess_data does
possible_label_cols = [c for c in df2.columns if c.lower() in ("label", "attack", "class", "isattack", "is_malicious")]
print(f'2. Found label cols: {possible_label_cols}')

if possible_label_cols and 'fraud' not in df2.columns:
    lab = possible_label_cols[0]
    print(f'3. Using label column: "{lab}"')
    print(f'   dtype: {df2[lab].dtype}')
    print(f'   Sample values: {df2[lab].head(3).tolist()}')
    
    try:
        if df2[lab].dtype == object or df2[lab].dtype.name.startswith('string'):
            print(f'4. Creating fraud from string labels...')
            df_temp = df2[lab].astype(str)
            fraud_benign = df_temp.str.strip().str.upper().eq('BENIGN').astype(int)
            df2['fraud'] = (1 - fraud_benign)  # Invert
            print(f'5. Fraud distribution: {df2["fraud"].value_counts().to_dict()}')
        else:
            print(f'4. Labels are NOT object type (type={df2[lab].dtype}), creating from numeric...')
            df2['fraud'] = (df2[lab] != 0).astype(int)
            print(f'5. Fraud distribution: {df2["fraud"].value_counts().to_dict()}')
    except Exception as e:
        print(f'ERROR creating fraud column: {e}')
        import traceback
        traceback.print_exc()
else:
    print(f'3. ERROR: Could not create fraud column! possible_label_cols={possible_label_cols}, has fraud="fraud" in df2.columns')

# Now test what the SMOTE balancing does
print('\n\nTesting SMOTE balancing:')
y = df2['fraud'].copy()
print(f'Before balance: {y.value_counts().to_dict()}')

if (y == 1).sum() > 0 and (y == 0).sum() > 0 and len(y) >100:
    from imblearn.over_sampling import SMOTE
    from modules.data_loader import preprocess_data
    print('Applying SMOTE...')
    # Get X for SMOTE
    X = df2.drop([lab, 'fraud'], axis=1)
    #Just get numeric cols
    X = X.select_dtypes(include=['number'])
    try:
        smote = SMOTE(
            random_state=DATA_CONFIG["smote_random_state"],
            k_neighbors=min(5, (y == 1).sum() - 1)
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f'After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}')
    except Exception as e:
        print(f'ERROR in SMOTE: {e}')
