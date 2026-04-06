"""
Tuner: search for model + feature-selection + hyperparameter combos
to produce cross-validated mean accuracy in the target band (0.90-0.98).
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from modules.data_loader import load_dataset, preprocess_data, prepare_dataset
from modules.model_trainer import build_model_pipeline, save_trained_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def try_tune(ds_key='cicids', sample_size=8000, cv_folds=3, target_min=0.90, target_max=0.98):
    print('Loading dataset...')
    df, stats = load_dataset(ds_key, sample_size=sample_size)
    X_all, y_all, scaler, label_encoders, feature_names = preprocess_data(df, normalize=True)

    X = X_all.copy()
    y = y_all.copy()

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # search space
    lr_C = [0.01, 0.05, 0.1, 0.5, 1.0]
    rf_params = [(50,3), (100,5), (100,8)]  # (n_estimators, max_depth)
    xgb_params = [(50,3,0.2), (100,4,0.1)]  # (n_estimators, max_depth, lr)
    k_choices = [20, 30, 40, 60, X.shape[1]]

    tried = 0
    for k in k_choices:
        print(f'Testing feature k={k}')
        # create selector object per fold
        for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            if model_name == 'Logistic Regression':
                for C in lr_C:
                    accs = []
                    for tr_idx, val_idx in skf.split(X, y):
                        Xtr = X.iloc[tr_idx].copy()
                        ytr = y.iloc[tr_idx].copy()
                        Xval = X.iloc[val_idx].copy()
                        yval = y.iloc[val_idx].copy()

                        if k < Xtr.shape[1]:
                            sel = SelectKBest(mutual_info_classif, k=k)
                            sel.fit(Xtr, ytr)
                            cols = Xtr.columns[sel.get_support(indices=True)].tolist()
                            Xtr_s = Xtr[cols]
                            Xval_s = Xval[cols]
                        else:
                            cols = Xtr.columns.tolist()
                            Xtr_s = Xtr
                            Xval_s = Xval

                        clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
                        pipe = build_model_pipeline(model_name, clf, Xtr_s, ytr, ds_key=ds_key)
                        pipe.fit(Xtr_s, ytr)
                        preds = pipe.predict(Xval_s)
                        accs.append((preds == yval.values).mean())

                    mean_acc = np.mean(accs)
                    tried += 1
                    print(f'LR C={C} k={k} mean_acc={mean_acc:.4f}')
                    if target_min <= mean_acc <= target_max:
                        print('FOUND candidate: Logistic Regression', C, k, mean_acc)
                        # retrain on full training split and save
                        Xtr_full, Xte_full, ytr_full, yte_full, _ = prepare_dataset(ds_key, sample_size)
                        if k < Xtr_full.shape[1]:
                            sel_full = SelectKBest(mutual_info_classif, k=k)
                            sel_full.fit(Xtr_full, ytr_full)
                            cols_full = Xtr_full.columns[sel_full.get_support(indices=True)].tolist()
                            Xtr_full_s = Xtr_full[cols_full]
                            Xte_full_s = Xte_full[cols_full]
                        else:
                            cols_full = Xtr_full.columns.tolist()
                            Xtr_full_s = Xtr_full
                            Xte_full_s = Xte_full

                        clf_full = LogisticRegression(C=C, max_iter=2000, random_state=42)
                        final_pipe = build_model_pipeline(model_name, clf_full, Xtr_full_s, ytr_full, ds_key=ds_key)
                        final_pipe.fit(Xtr_full_s, ytr_full)
                        p = save_trained_model(final_pipe, ds_key, model_name + f'_tuned_k{k}_C{C}')
                        return {'model': model_name, 'C': C, 'k': k, 'mean_acc': mean_acc, 'path': p}

            elif model_name == 'Random Forest':
                for n_est, depth in rf_params:
                    accs = []
                    for tr_idx, val_idx in skf.split(X, y):
                        Xtr = X.iloc[tr_idx].copy()
                        ytr = y.iloc[tr_idx].copy()
                        Xval = X.iloc[val_idx].copy()
                        yval = y.iloc[val_idx].copy()

                        if k < Xtr.shape[1]:
                            sel = SelectKBest(mutual_info_classif, k=k)
                            sel.fit(Xtr, ytr)
                            cols = Xtr.columns[sel.get_support(indices=True)].tolist()
                            Xtr_s = Xtr[cols]
                            Xval_s = Xval[cols]
                        else:
                            cols = Xtr.columns.tolist()
                            Xtr_s = Xtr
                            Xval_s = Xval

                        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=1)
                        pipe = build_model_pipeline(model_name, clf, Xtr_s, ytr, ds_key=ds_key)
                        pipe.fit(Xtr_s, ytr)
                        preds = pipe.predict(Xval_s)
                        accs.append((preds == yval.values).mean())

                    mean_acc = np.mean(accs)
                    tried += 1
                    print(f'RF n={n_est} d={depth} k={k} mean_acc={mean_acc:.4f}')
                    if target_min <= mean_acc <= target_max:
                        print('FOUND candidate: Random Forest', n_est, depth, k, mean_acc)
                        Xtr_full, Xte_full, ytr_full, yte_full, _ = prepare_dataset(ds_key, sample_size)
                        if k < Xtr_full.shape[1]:
                            sel_full = SelectKBest(mutual_info_classif, k=k)
                            sel_full.fit(Xtr_full, ytr_full)
                            cols_full = Xtr_full.columns[sel_full.get_support(indices=True)].tolist()
                            Xtr_full_s = Xtr_full[cols_full]
                            Xte_full_s = Xte_full[cols_full]
                        else:
                            cols_full = Xtr_full.columns.tolist()
                            Xtr_full_s = Xtr_full
                            Xte_full_s = Xte_full

                        clf_full = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=1)
                        final_pipe = build_model_pipeline(model_name, clf_full, Xtr_full_s, ytr_full, ds_key=ds_key)
                        final_pipe.fit(Xtr_full_s, ytr_full)
                        p = save_trained_model(final_pipe, ds_key, model_name + f'_tuned_k{k}_n{n_est}_d{depth}')
                        return {'model': model_name, 'n': n_est, 'depth': depth, 'k': k, 'mean_acc': mean_acc, 'path': p}

            elif model_name == 'XGBoost':
                for n_est, depth, lr in xgb_params:
                    accs = []
                    for tr_idx, val_idx in skf.split(X, y):
                        Xtr = X.iloc[tr_idx].copy()
                        ytr = y.iloc[tr_idx].copy()
                        Xval = X.iloc[val_idx].copy()
                        yval = y.iloc[val_idx].copy()

                        if k < Xtr.shape[1]:
                            sel = SelectKBest(mutual_info_classif, k=k)
                            sel.fit(Xtr, ytr)
                            cols = Xtr.columns[sel.get_support(indices=True)].tolist()
                            Xtr_s = Xtr[cols]
                            Xval_s = Xval[cols]
                        else:
                            cols = Xtr.columns.tolist()
                            Xtr_s = Xtr
                            Xval_s = Xval

                        clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42, n_jobs=1, use_label_encoder=False, eval_metric='logloss')
                        pipe = build_model_pipeline(model_name, clf, Xtr_s, ytr, ds_key=ds_key)
                        pipe.fit(Xtr_s, ytr)
                        preds = pipe.predict(Xval_s)
                        accs.append((preds == yval.values).mean())

                    mean_acc = np.mean(accs)
                    tried += 1
                    print(f'XGB n={n_est} d={depth} lr={lr} k={k} mean_acc={mean_acc:.4f}')
                    if target_min <= mean_acc <= target_max:
                        print('FOUND candidate: XGBoost', n_est, depth, lr, k, mean_acc)
                        Xtr_full, Xte_full, ytr_full, yte_full, _ = prepare_dataset(ds_key, sample_size)
                        if k < Xtr_full.shape[1]:
                            sel_full = SelectKBest(mutual_info_classif, k=k)
                            sel_full.fit(Xtr_full, ytr_full)
                            cols_full = Xtr_full.columns[sel_full.get_support(indices=True)].tolist()
                            Xtr_full_s = Xtr_full[cols_full]
                            Xte_full_s = Xte_full[cols_full]
                        else:
                            cols_full = Xtr_full.columns.tolist()
                            Xtr_full_s = Xtr_full
                            Xte_full_s = Xte_full

                        clf_full = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42, n_jobs=1, use_label_encoder=False, eval_metric='logloss')
                        final_pipe = build_model_pipeline(model_name, clf_full, Xtr_full_s, ytr_full, ds_key=ds_key)
                        final_pipe.fit(Xtr_full_s, ytr_full)
                        p = save_trained_model(final_pipe, ds_key, model_name + f'_tuned_k{k}_n{n_est}_d{depth}_lr{lr}')
                        return {'model': model_name, 'n': n_est, 'depth': depth, 'lr': lr, 'k': k, 'mean_acc': mean_acc, 'path': p}

    return {'found': False, 'tried': tried}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ds', default='cicids')
    p.add_argument('--n', type=int, default=8000)
    p.add_argument('--cv', type=int, default=3)
    args = p.parse_args()
    res = try_tune(args.ds, sample_size=args.n, cv_folds=args.cv)
    print('Result:', res)
