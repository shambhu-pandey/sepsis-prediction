"""
Local evaluation runner: loads a dataset, preprocesses, splits, trains models, prints metrics.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from config import DATASET_META, DATA_CONFIG
from modules.data_loader import preprocess_data, split_data, load_upi_csv, load_cicids2017_csv
from modules.model_trainer import train_and_evaluate_all


def run(ds_key='cicids', sample_rows=5000):
    print(f"Running local eval for: {ds_key}")
    print('Loading dataset (with stratified sampling if requested)...')
    from modules.data_loader import load_dataset
    try:
        df, stats = load_dataset(ds_key, sample_size=sample_rows)
    except Exception as e:
        print('Error loading dataset via load_dataset():', e)
        return

    print(f'Loaded {len(df)} rows; sampling_used={stats.get("sampling_used")}')

    print('Preprocessing...')
    X, y, scaler, label_encoders, feature_names = preprocess_data(df, normalize=True)
    print('Preprocess done. X shape:', X.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print('Split:', X_train.shape, X_test.shape)

    print('Training models...')
    results, models, best, timings = train_and_evaluate_all(X_train, y_train, X_test, y_test, ds_key=ds_key)
    print('\nResults:')
    for k, v in results.items():
        print(k, v)
    print('\nBest:', best)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ds', default='cicids')
    p.add_argument('--n', type=int, default=5000)
    args = p.parse_args()
    run(args.ds, args.n)
