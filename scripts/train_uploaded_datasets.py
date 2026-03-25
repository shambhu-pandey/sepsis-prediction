"""
Train models on datasets placed in the `data/` folder and save trained models.

This script looks for PaySim and CICIDS CSV files and trains all models using
the existing pipeline (preprocess_data, train_all_models). It saves models
with dataset-specific prefixes so the dashboard can load them later.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_SOURCES, DATA_CONFIG
from modules.data_loader import load_dataset, preprocess_data, split_data
from modules.model_trainer import train_all_models
from modules.utils import save_model


def find_dataset_files(data_dir="data"):
    p = Path(data_dir)
    files = list(p.glob("*.csv"))
    return files


def dataset_key_for_filename(fn: str):
    name = fn.lower()
    if "paysim" in name:
        return "paysim"
    if "cicids" in name or "cicids2017" in name:
        return "cicids"
    if "upi" in name:
        return "upi"
    return "synthetic"


def main():
    data_dir = ROOT / "data"
    files = find_dataset_files(data_dir)
    if not files:
        print("No CSV files found in data/. Place paysim.csv or cicids.csv there.")
        return

    for f in files:
        print(f"\n=== Processing {f.name} ===")
        key = dataset_key_for_filename(f.name)
        # Load dataset via loader (it will read local path)
        df = load_dataset(key, file_path=str(f), sample_size=DATA_CONFIG.get("sample_size", 10000))
        if df is None or df.empty:
            print(f"Failed to load {f.name}. Skipping.")
            continue

        try:
            X, y, scaler, label_encoders, feature_names, stats = preprocess_data(
                df, handle_missing=True, normalize=True, balance=True, sample_size=DATA_CONFIG.get("sample_size", 10000)
            )
        except Exception as e:
            print(f"Preprocessing failed for {f.name}: {e}")
            continue

        X_train, X_test, y_train, y_test = split_data(X, y)

        print("Training models (this may take a few minutes)...")
        results, models = train_all_models(X_train, y_train, X_test, y_test)

        # Save each model with dataset suffix
        for model_name, model_data in models.items():
            prefix = model_name.replace(" ", "_").lower() + "_" + key
            save_path = save_model(model_data["model"], prefix)
            print(f"Saved {model_name} -> {save_path}")

        print("Results:")
        for m, metrics in results.items():
            print(f" - {m}: ROC-AUC={metrics.get('ROC-AUC'):.4f}, F1={metrics.get('F1-Score'):.4f}")


if __name__ == "__main__":
    main()
