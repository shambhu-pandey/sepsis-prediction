"""
Targeted correlation check for IEEE and CICIDS datasets.
"""
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from modules.data_loader import _read_csv, _find_target, _binarize_target
from config import DATASET_META, RAW_FEATURES_BY_DATASET

for key in ["ieee", "cicids", "paysim"]:
    meta = DATASET_META[key]
    path = meta["file"]
    if not os.path.exists(path):
        for alt in meta.get("alt_files", []):
            if os.path.exists(alt):
                path = alt; break
    if not os.path.exists(path):
        print(f"SKIP {key}: file not found"); continue

    df = _read_csv(path)
    if df is None: continue
    df.columns = df.columns.str.strip()
    target_col = _find_target(df, meta["target_col"])
    if not target_col: continue
    df = _binarize_target(df, target_col)

    # Sample 10k for speed
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    y = df["fraud"]
    X = df.drop(columns=["fraud"], errors="ignore")

    required = RAW_FEATURES_BY_DATASET.get(key, [])
    X = X[[c for c in required if c in X.columns]]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    num_X = X.select_dtypes(include=[np.number])
    temp = num_X.copy()
    temp["target"] = y.values
    corrs = temp.corr()["target"].abs().sort_values(ascending=False)
    print(f"\n=== {key.upper()} absolute correlations ===")
    print(corrs.head(10).to_string())
    perfect = corrs[corrs > 0.9].index.tolist()
    if "target" in perfect: perfect.remove("target")
    if perfect:
        print(f"  ⚠️  HIGH CORRELATION COLS: {perfect}")
    else:
        print(f"  ✅  All correlations are realistic (< 0.9)")
