"""
Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
import pytz
import streamlit as st
import os
from config import DATA_CONFIG, NUMERIC_FEATURES, CATEGORICAL_FEATURES, UPI_CONFIG


@st.cache_data
def generate_upi_dataset(num_records=None, fraud_rate=None):
    """
    Generate synthetic UPI dataset with realistic fields.
    
    Args:
        num_records: Number of records to generate
        fraud_rate: Percentage of fraudulent transactions
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    if num_records is None:
        num_records = UPI_CONFIG["num_records"]
    if fraud_rate is None:
        fraud_rate = UPI_CONFIG["fraud_rate"]
    
    transaction_types = UPI_CONFIG["transaction_types"]
    device_types = ["Android", "iOS", "Web"]
    
    np.random.seed(42)
    
    # Generate transaction data
    data = {
        "transaction_id": [f"TXN{i:07d}" for i in range(num_records)],
        "sender": [f"USER{np.random.randint(1000, 999999)}" for _ in range(num_records)],
        "receiver": [f"USER{np.random.randint(1000, 999999)}" for _ in range(num_records)],
        "amount": np.random.gamma(shape=2, scale=1000, size=num_records),
        "device_id": [f"DEV{np.random.randint(100000, 999999)}" for _ in range(num_records)],
        "sender_device_type": np.random.choice(device_types, size=num_records),
        "receiver_device_type": np.random.choice(device_types, size=num_records),
        "location": [f"Location_{np.random.randint(1, 100)}" for _ in range(num_records)],
        "time": [datetime.now() - timedelta(days=np.random.randint(0, 365), 
                                           hours=np.random.randint(0, 24),
                                           minutes=np.random.randint(0, 60)) 
                for _ in range(num_records)],
        "transaction_type": np.random.choice(transaction_types, size=num_records),
        "transaction_count_24h": np.random.poisson(lam=5, size=num_records),
        "location_change_indicator": np.random.binomial(n=1, p=0.1, size=num_records),
        # use float so we can assign non-integer values without dtype errors
        "device_age_days": np.random.gamma(shape=2, scale=365, size=num_records),
    }
    
    df = pd.DataFrame(data)
    
    # Extract time features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.weekday
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    
    # Generate fraud labels
    num_fraud = int(num_records * fraud_rate)
    fraud_indices = np.random.choice(num_records, size=num_fraud, replace=False)
    df["fraud"] = 0
    # add a reason column to help users understand why a record was flagged
    df["fraud_reason"] = "legitimate"
    
    # Add fraud patterns
    for idx in fraud_indices:
        pattern = np.random.choice([0, 1, 2])
        reason = ""
        if pattern == 0:  # Unusual amount
            df.loc[idx, "amount"] = np.random.uniform(40000, 50000)
            reason = "High transaction amount"
        elif pattern == 1:  # New device + location change
            # allow fractional age, keep float dtype
            df.loc[idx, "device_age_days"] = float(np.random.uniform(0, 30))
            df.loc[idx, "location_change_indicator"] = 1
            reason = "New device and location change"
        else:  # Unusual time + high transaction count
            df.loc[idx, "hour"] = np.random.choice([2, 3, 4])
            df.loc[idx, "transaction_count_24h"] = np.random.randint(15, 30)
            reason = "Odd hour/High frequency"
        
        df.loc[idx, "fraud"] = 1
        df.loc[idx, "fraud_reason"] = reason
    
    # Clip amount to reasonable range
    df["amount"] = df["amount"].clip(lower=10, upper=50000)
    
    # Remove time column for model training
    df = df.drop("time", axis=1)
    
    return df



# -----------------------------------------------------------------------------
# External dataset loading utilities
# -----------------------------------------------------------------------------

def _read_csv_safely(path):
    """Safely read a CSV file and return a DataFrame or None.

    Args:
        path (str): Path to CSV file.

    Returns:
        pd.DataFrame or None
    """
    try:
        if path is None:
            return None
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        return df
    except Exception as e:
        try:
            st.warning(f"Could not read CSV at {path}: {e}")
        except Exception:
            pass
        return None
    
    


@st.cache_data
def load_upi_csv(file_path):
    """
    Load a UPI-formatted CSV file provided by the user.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame
    """
    return _read_csv_safely(file_path)


@st.cache_data
def load_cicids2017_csv(file_path):
    """
    Load CICIDS2017 dataset from a CSV file path.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame
    """
    return _read_csv_safely(file_path)


@st.cache_data
def load_dataset(name="synthetic", file_path=None, sample_size=None):
    """
    Unified loader for all supported data sources.

    Args:
        name (str): One of 'synthetic', 'paysim', 'upi', 'cicids'.
        file_path (str): Optional local CSV path (for upi, cicids, or paysim).
        sample_size (int): Maximum number of rows to sample.

    Returns:
        pd.DataFrame
    """
    name = name.lower()
    df = pd.DataFrame()

    if name == "synthetic":
        # use sample_size as num_records if provided
        num = sample_size if sample_size is not None else UPI_CONFIG["num_records"]
        rate = UPI_CONFIG.get("fraud_rate", 0.02)
        df = generate_upi_dataset(num_records=num, fraud_rate=rate)
    elif name in ["paysim", "paysim_kaggle"]:
        # Try local file first (preferred), then fallback to synthetic data
        if file_path:
            df = _read_csv_safely(file_path)
            if df is None or df.empty:
                # Try alternative local files
                alt_paths = ["data/paysim_clean.csv", "data/paysim.csv"]
                for alt_path in alt_paths:
                    df = _read_csv_safely(alt_path)
                    if df is not None and not df.empty:
                        break
        else:
            # No file_path provided, try local data folder
            local_paths = ["data/paysim_clean.csv", "data/paysim.csv"]
            df = None
            for path in local_paths:
                df = _read_csv_safely(path)
                if df is not None and not df.empty:
                    break
            
            # If no local files, fall back to synthetic data
            if df is None or df.empty:
                st.info("No local PaySim data found. Using synthetic dataset.")
                df = generate_upi_dataset(num_records=sample_size or UPI_CONFIG["num_records"], fraud_rate=UPI_CONFIG.get("fraud_rate", 0.02))
    elif name in ["upi", "upi_csv"]:
        if file_path:
            df = load_upi_csv(file_path)
        else:
            st.warning("No UPI CSV file specified; falling back to synthetic data.")
            df = generate_upi_dataset(num_records=sample_size or UPI_CONFIG["num_records"], fraud_rate=UPI_CONFIG.get("fraud_rate", 0.02))
    elif name in ["cicids", "cicids2017"]:
        # Try local file first (preferred), then fallback to synthetic data
        if file_path:
            df = load_cicids2017_csv(file_path)
            if df is None or df.empty:
                # Try alternative local files
                alt_paths = ["data/cicids_clean.csv", "data/cicids.csv"]
                for alt_path in alt_paths:
                    df = _read_csv_safely(alt_path)
                    if df is not None and not df.empty:
                        break
        else:
            # No file_path provided, try local data folder
            local_paths = ["data/cicids_clean.csv", "data/cicids.csv"]
            df = None
            for path in local_paths:
                df = _read_csv_safely(path)
                if df is not None and not df.empty:
                    break
            
            # If no local files, fall back to synthetic data
            if df is None or df.empty:
                st.info("No local CICIDS data found. Using synthetic dataset.")
                df = generate_upi_dataset(num_records=sample_size or UPI_CONFIG["num_records"], fraud_rate=UPI_CONFIG.get("fraud_rate", 0.02))
    else:
        st.warning(f"Unknown dataset '{name}', using synthetic UPI dataset.")
        df = generate_upi_dataset(num_records=sample_size or UPI_CONFIG["num_records"], fraud_rate=UPI_CONFIG.get("fraud_rate", 0.02))

    # sample down if requested and df large
    if sample_size and not df.empty and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=DATA_CONFIG["random_state"])

    return df


def preprocess_data(df, handle_missing=True, normalize=True, balance=True, sample_size=None):
    """
    Preprocess data: handle missing values, normalize, and balance.
    
    Args:
        df: Input DataFrame
        handle_missing: Whether to handle missing values
        normalize: Whether to normalize features
        balance: Whether to balance fraud vs non-fraud
        sample_size: Optional sample size for reduced computation
    
    Returns:
        tuple: (X_processed, y, scaler, label_encoders, feature_names, stats)
            stats: dict containing original counts before sampling/balancing
    """
    df = df.copy()
    # normalize column names (strip whitespace) to make label detection robust
    try:
        df.columns = df.columns.str.strip()
    except Exception:
        pass

    # If dataset uses a 'Label' or similar column (e.g., CICIDS), map to binary 'fraud'
    # common patterns: 'Label' with 'BENIGN' or attack names, or other class columns
    possible_label_cols = [c for c in df.columns if c.lower() in ("label", "attack", "class", "isattack", "is_malicious")] 
    if possible_label_cols and 'fraud' not in df.columns:
        lab = possible_label_cols[0]
        try:
            # Check if column contains strings (object, 'string', or 'str' dtype)
            is_string_dtype = (df[lab].dtype == object or 
                             df[lab].dtype.name in ('string', 'str', 'object') or
                             str(df[lab].dtype).startswith(('string', 'object')))
            if is_string_dtype:
                # More robust string comparison
                df_temp = df[lab].astype(str)
                df['fraud'] = df_temp.str.strip().str.upper().eq('BENIGN').astype(int)
                df['fraud'] = (1 - df['fraud'])  # Invert so BENIGN=0, others=1
            else:
                # numeric labels: assume non-zero indicates attack
                df['fraud'] = (df[lab] != 0).astype(int)
        except Exception as e:
            pass
    
    # original statistics before any sampling or balancing
    orig_total = len(df)
    fraud_col = None
    for col in ["fraud", "isFraud", "is_fraud", "Class"]:
        if col in df.columns:
            fraud_col = col
            break
    if fraud_col is not None:
        orig_fraud = (df[fraud_col] == 1).sum()
    else:
        orig_fraud = 0
    
    # Sample if needed
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=DATA_CONFIG["random_state"])
    
    # Identify fraud column (again in case sampling removed it?)
    if fraud_col is None:
        raise ValueError("Could not identify fraud column in dataset")
    
    y = df[fraud_col].copy()
    X = df.drop([fraud_col], axis=1)
    
    # Auto-detect numeric and categorical columns if config features don't match
    # (for datasets like CICIDS with different column names)
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    
    # If we have matching features from config, use those; otherwise use auto-detected
    matching_numeric = [col for col in NUMERIC_FEATURES if col in X.columns]
    matching_categorical = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    
    if matching_numeric or matching_categorical:
        # UPI-style dataset with matching config features
        relevant_cols = matching_numeric + matching_categorical
    else:
        # External dataset (CICIDS, PaySim, etc.) - use all numeric columns
        relevant_cols = numeric_cols
    
    X = X[relevant_cols]
    
    # Handle infinite values (replace with NaN so fillna handles them)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values
    if handle_missing:
        X = X.fillna(X.mean(numeric_only=True))
    
    # Label encoding for categorical features (only those present in X)
    label_encoders = {}
    present_categorical = [col for col in categorical_cols if col in X.columns]
    for col in present_categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    # Balance dataset using SMOTE
    if balance and len(y) > 100:
        try:
            # Only apply if fraud exists
            if (y == 1).sum() > 0 and (y == 0).sum() > 0:
                smote = SMOTE(
                    random_state=DATA_CONFIG["smote_random_state"],
                    k_neighbors=min(5, (y == 1).sum() - 1)
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled, name=y.name)
        except Exception as e:
            st.warning(f"Could not apply SMOTE: {e}")
    
    stats = {
        "original_total": orig_total,
        "original_fraud": orig_fraud,
        "post_balance_total": len(y),
        "post_balance_fraud": (y == 1).sum()
    }
    return X, y, scaler, label_encoders, X.columns.tolist(), stats


def split_data(X, y, test_size=None, random_state=None):
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Test set size
        random_state: Random state
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if test_size is None:
        test_size = DATA_CONFIG["test_size"]
    if random_state is None:
        random_state = DATA_CONFIG["random_state"]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# Sample fraud and normal transaction examples for demonstration
SAMPLE_TRANSACTIONS = {
    "fraud": [
        {
            "name": "High Amount Fraud",
            "description": "Transaction with unusually high amount",
            "reason": "High transaction amount is a common fraud indicator"
        },
        {
            "name": "New Device + Location Change",
            "description": "Transaction from a new device with a location change",
            "reason": "New device and location change indicates potential account takeover"
        },
        {
            "name": "Odd Hours + High Frequency",
            "description": "Multiple transactions at unusual hours",
            "reason": "Transaction at odd hours with high frequency is suspicious"
        },
        {
            "name": "Large Amount + New Device",
            "description": "Large transaction from brand new device",
            "reason": "Large amount from new device with location change"
        }
    ],
    "normal": [
        {
            "name": "Regular P2P Transfer",
            "description": "Normal peer-to-peer transfer",
            "reason": "Regular transaction with normal amount and device"
        },
        {
            "name": "Small Merchant Payment",
            "description": "Small payment to merchant from known device",
            "reason": "Small amount from old device with no red flags"
        },
        {
            "name": "Recharge Transaction",
            "description": "Normal mobile recharge",
            "reason": "Standard recharge with known device"
        },
        {
            "name": "Bill Payment - Regular Hours",
            "description": "Bill payment during business hours",
            "reason": "Normal bill payment during regular hours"
        }
    ]
}


# Sample transactions for CICIDS network attack detection
CICIDS_SAMPLE_TRANSACTIONS = {
    "attack": [
        {
            "name": "DDoS Attack Pattern",
            "description": "High volume traffic from multiple sources",
            "reason": "Extremely high packet count and byte volume indicates DDoS attack"
        },
        {
            "name": "Port Scanning",
            "description": "Systematic access to multiple ports",
            "reason": "Low data transfer with systematic port access suggests reconnaissance"
        },
        {
            "name": "Brute Force Attack",
            "description": "Multiple connection attempts",
            "reason": "Repeated connection attempts to same port indicates brute force"
        },
        {
            "name": "Botnet Traffic",
            "description": "Persistent suspicious connections",
            "reason": "Long duration with regular data exfiltration pattern"
        }
    ],
    "normal": [
        {
            "name": "Normal Web Browsing",
            "description": "Standard HTTP traffic",
            "reason": "Normal HTTP traffic pattern with reasonable volume"
        },
        {
            "name": "DNS Query",
            "description": "Standard DNS lookup",
            "reason": "Typical DNS query pattern"
        },
        {
            "name": "Secure HTTPS Session",
            "description": "Normal encrypted web traffic",
            "reason": "Standard secure web session"
        },
        {
            "name": "Email Traffic",
            "description": "Normal email IMAP connection",
            "reason": "Normal email retrieval pattern"
        }
    ]
}


def get_sample_transactions(transaction_type="all"):
    """
    Get sample transactions for demonstration.
    
    Args:
        transaction_type: "all", "fraud", "normal", "attack", or "normal"
    
    Returns:
        dict: Dictionary with fraud and/or normal transactions
    """
    if transaction_type == "all":
        return SAMPLE_TRANSACTIONS
    elif transaction_type in SAMPLE_TRANSACTIONS:
        return {transaction_type: SAMPLE_TRANSACTIONS[transaction_type]}
    else:
        return SAMPLE_TRANSACTIONS


def get_cicids_sample_transactions(transaction_type="all"):
    """
    Get sample CICIDS network traffic for demonstration.
    
    Args:
        transaction_type: "all", "attack", or "normal"
    
    Returns:
        dict: Dictionary with attack and/or normal network traffic
    """
    if transaction_type == "all":
        return CICIDS_SAMPLE_TRANSACTIONS
    elif transaction_type in CICIDS_SAMPLE_TRANSACTIONS:
        return {transaction_type: CICIDS_SAMPLE_TRANSACTIONS[transaction_type]}
    else:
        return CICIDS_SAMPLE_TRANSACTIONS


def validate_and_prepare_transaction(transaction_dict, scaler=None, label_encoders=None, feature_names=None):
    """
    Validate and prepare a single transaction for prediction.
    
    Args:
        transaction_dict: Transaction details
        scaler: Fitted scaler
        label_encoders: Fitted label encoders
        feature_names: Expected feature names
    
    Returns:
        tuple: (is_valid, processed_data or error_message, original_data)
    """
    try:
        from modules.utils import validate_transaction_data, get_transaction_time_features
        
        is_valid, message = validate_transaction_data(transaction_dict)
        if not is_valid:
            return False, message, None
        
        # Extract time features
        time_obj = datetime.fromisoformat(transaction_dict["time"])
        time_features = get_transaction_time_features(time_obj)
        
        # Build feature vector
        transaction_features = {
            "amount": transaction_dict.get("amount", 0),
            "hour": time_features["hour"],
            "day_of_week": time_features["day_of_week"],
            "device_age_days": transaction_dict.get("device_age_days", 365),
            "transaction_count_24h": transaction_dict.get("transaction_count_24h", 1),
            "location_change_indicator": transaction_dict.get("location_change_indicator", 0),
            "transaction_type": transaction_dict.get("transaction_type", "P2P"),
            "sender_device_type": transaction_dict.get("sender_device_type", "Android"),
            "receiver_device_type": transaction_dict.get("receiver_device_type", "Android"),
        }
        
        # Create DataFrame
        X_transaction = pd.DataFrame([transaction_features])
        
        # Encode categorical features
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in X_transaction.columns:
                    try:
                        X_transaction[col] = encoder.transform(X_transaction[col].astype(str))
                    except Exception:
                        # If unknown label, use 0
                        X_transaction[col] = 0
        
        # Select only required features
        if feature_names:
            missing_cols = set(feature_names) - set(X_transaction.columns)
            for col in missing_cols:
                X_transaction[col] = 0
            X_transaction = X_transaction[feature_names]
        
        # Scale if scaler provided
        if scaler:
            X_transaction = pd.DataFrame(
                scaler.transform(X_transaction),
                columns=X_transaction.columns
            )
        
        return True, X_transaction, transaction_dict
    
    except Exception as e:
        return False, f"Error preparing transaction: {str(e)}", None
