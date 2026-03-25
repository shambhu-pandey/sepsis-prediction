import pandas as pd
from pathlib import Path

files = [Path('data/cicids.csv'), Path('data/cicids_clean.csv'), Path('data/paysim.csv'), Path('data/paysim_clean.csv')]
for f in files:
    try:
        if f.exists():
            df = pd.read_csv(f, nrows=0)
            print(f.name, 'columns:', df.columns.tolist())
        else:
            print(f.name, 'not found')
    except Exception as e:
        print(f.name, 'read error:', e)
