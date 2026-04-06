"""
Quick validation: test that the revised feature sets produce realistic accuracy (90-98%).
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_RUNNING"] = "1"  # suppress streamlit warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from modules.data_loader import _read_csv, _find_target, _binarize_target
from config import DATASET_META, RAW_FEATURES_BY_DATASET

def validate(key):
    meta = DATASET_META[key]
    path = meta["file"]
    if not os.path.exists(path):
        for alt in meta.get("alt_files", []):
            if os.path.exists(alt):
                path = alt; break
    if not os.path.exists(path):
        print(f"  SKIP {key}: file not found")
        return

    df = _read_csv(path)
    if df is None:
        print(f"  SKIP {key}: read failed"); return

    df.columns = df.columns.str.strip()
    target_col = _find_target(df, meta["target_col"])
    if not target_col:
        print(f"  SKIP {key}: target not found"); return

    df = _binarize_target(df, target_col)
    y = df["fraud"]
    X = df.drop(columns=["fraud"], errors="ignore")

    # Apply feature catalog
    required = RAW_FEATURES_BY_DATASET.get(key, [])
    missing = [c for c in required if c not in X.columns]
    if missing:
        print(f"  SKIP {key}: missing features {missing}"); return
    X = X[required]

    # Clean inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Sample
    cap = meta.get("sample_size") or 25000
    if len(df) > cap:
        idx = df.sample(n=cap, random_state=42).index
        X = X.loc[idx]; y = y.loc[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Simple pipeline
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    for name, clf in [("LR", LogisticRegression(max_iter=500, C=0.1)), ("RF", RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42))]:
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        flag = "✅" if 0.87 <= acc <= 0.99 else "❌ TOO PERFECT" if acc > 0.99 else "❌ TOO LOW"
        print(f"  {flag}  [{key}] {name}: acc={acc:.4f}")

print("=== Validation Run ===")
for ds in ["paysim", "banksim", "cicids", "ieee"]:
    print(f"\n--- {ds} ---")
    validate(ds)
print("\n=== Done ===")
