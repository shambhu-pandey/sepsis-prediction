"""
Data Loading and Preprocessing  -  Anti-Leakage Pipeline

Key design decisions:
  1. Leakage columns removed BEFORE anything else
  2. Data split BEFORE any fitting
  3. Encoders and scaler fit on training data ONLY
  4. Test data transformed with train-fit objects
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

from config import DATA_CONFIG, DATASET_META, LEAKAGE_COLS, RAW_FEATURES_BY_DATASET


def _read_csv(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
        
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    # Use standard load for reasonable files
    if file_size_mb < 200:
        # Try multiple encodings/engines for messy datasets like CICIDS
        for encoding in [None, 'latin1', 'unicode_escape']:
            try:
                # Use engine='c' first, then 'python' if it fails
                return pd.read_csv(path, low_memory=False, encoding=encoding)
            except Exception:
                try:
                    return pd.read_csv(path, low_memory=False, encoding=encoding, engine='python')
                except Exception:
                    continue
        print(f"  [ERROR] Failed to read {path} even with fallback encodings.")
        return None

    # Low-memory chunk ingestion for massive datasets (like PaySim 500MB+)
    # ... (rest of the code stays similar)
    try:
        chunks = []
        for chunk in pd.read_csv(path, low_memory=False, chunksize=100_000):
            target = _find_target(chunk, "isFraud") or _find_target(chunk, "fraud")
            if target and len(chunk) > 0:
                # Binarize early to detect
                if chunk[target].dtype == object or str(chunk[target].dtype).startswith("string"):
                    is_f = chunk[target].astype(str).str.strip().str.upper().ne("BENIGN")
                else:
                    is_f = chunk[target] != 0
                
                # Keep ALL frauds, but aggressively decimating Legitimate rows to save memory bounds
                fraud_df = chunk[is_f]
                legit_df = chunk[~is_f]
                # Downsample legits to fit inside typical memory bound
                frac = min(1.0, 10000 / max(1, len(legit_df)))
                legit_df = legit_df.sample(frac=frac, random_state=42)
                chunks.append(pd.concat([fraud_df, legit_df]))
            else:
                chunks.append(chunk.sample(frac=0.1, random_state=42))
        return pd.concat(chunks, ignore_index=True)
    except Exception:
        return None


def _resolve_file(key):
    meta = DATASET_META.get(key, {})
    for p in [meta.get("file", "")] + meta.get("alt_files", []):
        if p and os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def _stratified_sample(df, target, n, seed=42):
    if len(df) <= n:
        return df.copy()
    ratio = df[target].mean()
    n_pos = max(1, int(n * ratio))
    n_neg = n - n_pos
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if len(pos) < n_pos:
        pos_s = pos
        n_neg = n - len(pos_s)
    else:
        pos_s = pos.sample(n=n_pos, random_state=seed)
    neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=seed)
    return pd.concat([pos_s, neg_s]).sample(frac=1, random_state=seed).reset_index(drop=True)


def _find_target(df, expected):
    if expected in df.columns:
        return expected
    for c in df.columns:
        if c.lower() in ("fraud", "isfraud", "is_fraud", "label", "class"):
            return c
    return None


def _binarize_target(df, col):
    df = df.copy()
    data = df[col]
    if data.dtype == object or str(data.dtype).startswith("string"):
        df["fraud"] = data.astype(str).str.strip().str.upper().ne("BENIGN").astype(int)
    else:
        df["fraud"] = (data != 0).astype(int)
    if col != "fraud" and col in df.columns:
        df = df.drop(columns=[col])
    return df


# ---- Main loader ----

@st.cache_data(show_spinner=False)
def load_dataset(key, sample_size=None):
    meta = DATASET_META.get(key)
    if not meta:
        raise ValueError(f"Unknown dataset: {key}")
    path = _resolve_file(key)
    if not path:
        raise FileNotFoundError(f"CSV not found for {key}")
    df = _read_csv(path)
    if df is None or df.empty:
        raise ValueError(f"Empty CSV: {path}")

    df.columns = df.columns.str.strip()
    total = len(df)

    target = _find_target(df, meta["target_col"])
    if not target:
        raise ValueError(f"Target column not found for {key}")

    df = _binarize_target(df, target)
    orig_fraud = int(df["fraud"].sum())
    orig_ratio = orig_fraud / total

    cap = sample_size or meta.get("sample_size")
    sampled = False
    reason = ""
    if cap and len(df) > cap:
        df = _stratified_sample(df, "fraud", cap)
        sampled = True
        reason = (f"Dataset has {total:,} rows. Stratified sample of "
                  f"{len(df):,} rows preserving {orig_ratio:.4%} fraud ratio.")

    return df, {
        "total_rows_available": total,
        "total_features": len(df.columns),
        "rows_used": len(df),
        "original_fraud_count": orig_fraud,
        "original_fraud_ratio": orig_ratio,
        "sampled_fraud_count": int(df["fraud"].sum()),
        "sampled_fraud_ratio": float(df["fraud"].mean()),
        "sampling_used": sampled,
        "sampling_reason": reason,
        "file_path": path,
    }


from sklearn.base import BaseEstimator, TransformerMixin

class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline step.
    
    Key fix: derive features from raw columns BEFORE dropping parent ID columns.
    Only truly ID-like columns (customer, merchant, nameOrig, etc.) are dropped.
    Behavioral columns (step, amount) are KEPT — they are controlled upstream
    by RAW_FEATURES_BY_DATASET.
    """
    def __init__(self, raw_features=None):
        self.raw_features = raw_features
        self.expected_cols_ = None
        self.medians_ = {}

    def _engineer_and_clean(self, X):
        """Core feature engineering logic shared by fit() and transform()."""
        # 0. CLEAN INF/NAN EARLY (fixes CICIDS byte-rate columns)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # 1. DERIVED FEATURES — must happen BEFORE dropping parent columns
        # 💳 Merchant indicator (needs 'merchant' column)
        if "merchant" in X.columns:
            X["is_commercial"] = X["merchant"].astype(str).str.startswith('M').astype(int)

        # 🕒 Time features (needs 'step' or 'TransactionDT')
        if "step" in X.columns:
            X["hour_of_day"] = X["step"] % 24
            X["is_night"] = (X["hour_of_day"] < 6).astype(int)
        elif "TransactionDT" in X.columns:
            X["hour_of_day"] = (X["TransactionDT"] // 3600) % 24
            X["is_night"] = (X["hour_of_day"] < 6).astype(int)

        # 💰 Balance features (PaySim)
        if "amount" in X.columns and "oldbalanceOrg" in X.columns:
            X["balance_diff"] = X["oldbalanceOrg"] - X["amount"]
            X["amount_to_balance"] = X["amount"] / (X["oldbalanceOrg"] + 1)
            X["is_zero_balance"] = (X["oldbalanceOrg"] == 0).astype(int)
            X["is_large_transaction"] = (X["amount"] > 200000).astype(int)

        if "oldbalanceDest" in X.columns:
            X["is_empty_dest"] = (X["oldbalanceDest"] == 0).astype(int)

        if "type" in X.columns:
            X["is_suspicious_type"] = X["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

        # 2. DROP TRUE ID COLUMNS (customer, merchant, nameOrig, etc.)
        # NOTE: 'step' and 'amount' are NOT here — they are behavioral features.
        # They are controlled by RAW_FEATURES_BY_DATASET upstream.
        id_cols = [
            "transaction_id", "sender", "receiver", "device_id",
            "nameOrig", "nameDest", "customer", "merchant",
            "zipcodeOri", "zipMerchant",
        ]

        # All potential target name variants (defensive)
        target_cols = ["fraud", "isFraud", "is_fraud", "Class", "Label", "label"]
        drop_list = LEAKAGE_COLS + id_cols + target_cols

        for c in drop_list:
            if c in X.columns:
                X = X.drop(columns=[c])

        # 3. Coerce remaining object columns to numeric (except known categoricals)
        known_cats = {"type", "category", "gender", "age", "ProductCD",
                      "card4", "card6", "P_emaildomain", "DeviceType"}
        for c in X.columns:
            if X[c].dtype == 'object' and c not in known_cats:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        # 4. Handle NaN/inf in numeric columns
        num = X.select_dtypes(include=[np.number]).columns.tolist()
        X[num] = X[num].replace([np.inf, -np.inf], np.nan)

        return X, num

    def fit(self, X, y=None):
        X = self._prepare_raw_features(X)
        X, num = self._engineer_and_clean(X)

        # Store median for imputation at transform time
        for c in num:
            self.medians_[c] = X[c].median()

        X = X.fillna(0)
        self.expected_cols_ = X.columns.tolist()
        return self

    def _prepare_raw_features(self, X):
        X = X.copy()
        raw_features = getattr(self, "raw_features", None)
        if raw_features:
            missing = [c for c in raw_features if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required raw features: {missing}")
            X = X[raw_features]
        return X

    def transform(self, X):
        X = self._prepare_raw_features(X)
        X, num = self._engineer_and_clean(X)

        # Backfill missing columns expected from training
        if self.expected_cols_ is not None:
            for c in self.expected_cols_:
                if c not in X.columns:
                    X[c] = 0.0

        # Impute NaN with training medians
        for c in num:
            X[c] = X[c].fillna(self.medians_.get(c, 0.0))
        X = X.fillna(0)

        # Log-transform amount columns for better distribution
        amount_cols = [c for c in num if "amount" in c.lower() or "amt" in c.lower()]
        for col in amount_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))

        # Guarantee identical feature ordering output
        if self.expected_cols_ is not None:
            X = X[self.expected_cols_]

        return X

def extract_features(df):
    """
    Separate target (y) from features (X) with strict leakage assertion.
    """
    # All possible target column names
    leak_cols = ["fraud", "isFraud", "is_fraud", "Class", "Label", "label"]

    # Find the target column
    target_col = None
    for t in leak_cols:
        if t in df.columns:
            target_col = t
            break

    if target_col is not None:
        out = _binarize_target(df, target_col)
        y = out["fraud"].copy()
        X = out.drop(columns=["fraud"], errors="ignore")
    else:
        y = pd.Series([0] * len(df), index=df.index)
        X = df.copy()

    # Final defensive drop of all target-like columns
    X = X.drop(columns=[c for c in leak_cols if c in X.columns], errors="ignore")

    # 🔴 STRICT LEAKAGE ASSERTION
    for col in leak_cols:
        assert col not in X.columns, f"DATA LEAKAGE: target column '{col}' still present in features!"

    return X, y


# ---- Preprocessing (fit on train only) ----

def fit_preprocessor(X_train):
    X = X_train.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # One-Hot Encoding
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    feat_names = X.columns.tolist()
    scaler = StandardScaler()
    X_out = pd.DataFrame(scaler.fit_transform(X), columns=feat_names, index=X.index)
    return X_out, scaler, cat_cols, feat_names


def transform_data(X, scaler, cat_cols, feat_names):
    X = X.copy()
    
    # Apply OneHotEncoding matching train logic
    if cat_cols:
        X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns], drop_first=True)
        
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_names]

    return pd.DataFrame(scaler.transform(X), columns=feat_names, index=X.index)


# ---- Full pipeline ----

def prepare_dataset(key, sample_size=None):
    """
    Complete anti-leakage pipeline:
      load -> extract features -> split -> label noise -> fit on train -> transform test
    """
    df, stats = load_dataset(key, sample_size)
    X_raw, y = extract_features(df)
    required_features = RAW_FEATURES_BY_DATASET.get(key)
    if required_features:
        missing = [c for c in required_features if c not in X_raw.columns]
        if missing:
            raise ValueError(f"{key} dataset missing required features: {missing}")
        X_raw = X_raw[required_features].copy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=y,
    )

    # Let model_trainer handle scaling/encoding via imblearn Pipeline mapping safely
    return X_train_raw, X_test_raw, y_train, y_test, stats


# ---- Legacy compat ----

def preprocess_data(df, normalize=True):
    X, y = extract_features(df)
    # Defensive cleaning: replace infinite values and fill NaNs to avoid scaler errors
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    if normalize:
        Xp, sc, le, fn = fit_preprocessor(X)
        return Xp, y, sc, le, fn
    return X, y, None, {}, X.columns.tolist()

def split_data_tri(X, y, **kw):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, None, Xte, ytr, None, yte

def split_data(X, y, **kw):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def generate_upi_dataset(n=5000, fraud_rate=0.02):
    np.random.seed(42)
    df = pd.DataFrame({"amount": np.random.gamma(2,1000,n).clip(10,50000),
                        "hour": np.random.randint(0,24,n), "fraud": 0})
    df.loc[np.random.choice(n, int(n*fraud_rate), replace=False), "fraud"] = 1
    return df

def load_upi_csv(p): return _read_csv(p)
def load_cicids2017_csv(p): return _read_csv(p)
def get_sample_transactions(t="all"): return {"fraud":[],"normal":[]}
def get_cicids_sample_transactions(t="all"): return {"attack":[],"normal":[]}
def validate_and_prepare_transaction(*a,**k): return False,"Legacy",None
