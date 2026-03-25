import pandas as pd

df = pd.read_csv('data/cicids.csv')
df.columns = df.columns.str.strip()
print(f'Full dataset size: {len(df)}, Label dist: {df["Label"].value_counts().to_dict()}')

# Test different sample sizes with random_state=42
for size in [10000, 20000, 50000]:
    sample = df.sample(n=size, random_state=42)
    print(f'{size} sample: {sample["Label"].value_counts().to_dict()}')
