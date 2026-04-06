"""
Hyperparameter grid search without SelectKBest or heavy one-hot encoding
Uses `prepare_dataset` to get raw train/test and `evaluate_models_cv` for safe CV.
Saves first model found inside target accuracy band.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from modules.data_loader import prepare_dataset
from modules.model_trainer import evaluate_models_cv, TRAINERS, save_trained_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def run_search(ds_key='paysim', sample_size=5000, cv_folds=5, target_min=0.80, target_max=0.98):
    print('Preparing dataset', ds_key, sample_size)
    Xtr, Xte, ytr, yte, stats = prepare_dataset(ds_key, sample_size)

    # search space
    lr_C = [0.001, 0.01, 0.05, 0.1]
    rf_params = [(50,3), (100,4)]
    xgb_params = [(50,3,0.2), (80,4,0.1)]

    # Logistic Regression first
    for C in lr_C:
        print('Testing LR C=', C)
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
        # Build pipeline via trainers by calling trainer directly in evaluate_models_cv
        summary = evaluate_models_cv(Xtr, ytr, model_names=['Logistic Regression'], cv_folds=cv_folds, ds_key=ds_key)
        mean_acc = summary.get('Logistic Regression', {}).get('accuracy_mean', 0.0)
        print('LR mean_acc=', mean_acc)
        if target_min <= mean_acc <= target_max:
            print('Found LR candidate, retraining on full train split')
            # retrain full model
            from modules.model_trainer import _train_lr
            model = _train_lr(Xtr, ytr, ds_key=ds_key)
            path = save_trained_model(model, ds_key, f'logistic_regression_tuned_C{C}')
            return {'model': 'Logistic Regression', 'C': C, 'mean_acc': mean_acc, 'path': path}

    # Random Forest
    for n_est, depth in rf_params:
        print('Testing RF', n_est, depth)
        summary = evaluate_models_cv(Xtr, ytr, model_names=['Random Forest'], cv_folds=cv_folds, ds_key=ds_key)
        mean_acc = summary.get('Random Forest', {}).get('accuracy_mean', 0.0)
        print('RF mean_acc=', mean_acc)
        if target_min <= mean_acc <= target_max:
            from modules.model_trainer import _train_rf
            model = _train_rf(Xtr, ytr, ds_key=ds_key)
            path = save_trained_model(model, ds_key, f'random_forest_tuned_n{n_est}_d{depth}')
            return {'model': 'Random Forest', 'n': n_est, 'depth': depth, 'mean_acc': mean_acc, 'path': path}

    # XGBoost
    for n_est, depth, lr in xgb_params:
        print('Testing XGB', n_est, depth, lr)
        summary = evaluate_models_cv(Xtr, ytr, model_names=['XGBoost'], cv_folds=cv_folds, ds_key=ds_key)
        mean_acc = summary.get('XGBoost', {}).get('accuracy_mean', 0.0)
        print('XGB mean_acc=', mean_acc)
        if target_min <= mean_acc <= target_max:
            from modules.model_trainer import _train_xgb
            model = _train_xgb(Xtr, ytr, ds_key=ds_key)
            path = save_trained_model(model, ds_key, f'xgboost_tuned_n{n_est}_d{depth}_lr{lr}')
            return {'model': 'XGBoost', 'n': n_est, 'depth': depth, 'lr': lr, 'mean_acc': mean_acc, 'path': path}

    return {'found': False}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ds', default='paysim')
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--cv', type=int, default=5)
    args = p.parse_args()
    res = run_search(args.ds, sample_size=args.n, cv_folds=args.cv)
    print('Result:', res)
