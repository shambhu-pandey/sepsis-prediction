"""
Run CV across models and pick one whose mean accuracy is within target band (90%-98%).
Saves chosen model using save_trained_model().
"""
import sys
sys.path.insert(0, '.')

from config import DATASET_META
from modules.data_loader import load_dataset, preprocess_data
from modules.model_trainer import evaluate_models_cv, save_trained_model, train_and_evaluate_all


def choose_model_by_cv(ds_key='cicids', sample_size=10000, cv_folds=5, target_min=0.90, target_max=0.98):
    print(f"Running CV selector for {ds_key}")
    df, stats = load_dataset(ds_key, sample_size)
    X, y, scaler, label_encoders, feature_names = preprocess_data(df, normalize=True)

    print('Running cross-validation for each model...')
    summary = evaluate_models_cv(X, y, cv_folds=cv_folds, ds_key=ds_key)
    for k, v in summary.items():
        print(f"{k}: mean_acc={v['accuracy_mean']:.4f}, std={v['accuracy_std']:.4f}")

    # Find models within target band
    candidates = [(k, v['accuracy_mean']) for k, v in summary.items() if target_min <= v['accuracy_mean'] <= target_max]
    if not candidates:
        print('No candidates within target band. Picking closest below upper bound.')
        # pick model with mean accuracy closest to target_max but <= target_max, otherwise nearest overall
        diffs = [(k, abs(v['accuracy_mean'] - ((target_min+target_max)/2))) for k, v in summary.items()]
        candidates = sorted(diffs, key=lambda x: x[1])[:1]
        chosen = candidates[0][0]
    else:
        # pick candidate closest to midpoint of band
        midpoint = (target_min + target_max) / 2.0
        chosen = sorted(candidates, key=lambda x: abs(x[1] - midpoint))[0][0]

    print('Chosen model:', chosen)

    # Train chosen model on full training split and save
    # Use existing train_and_evaluate_all to fit models and retrieve chosen model
    print('Training chosen model on full split...')
    # Prepare a train/test split using data_loader.prepare_dataset
    from modules.data_loader import prepare_dataset
    Xtr, Xte, ytr, yte, stats2 = prepare_dataset(ds_key, sample_size)
    results, models, best, timings = train_and_evaluate_all(Xtr, ytr, Xte, yte, model_names=[chosen], ds_key=ds_key)
    model_obj = models.get(chosen)
    if model_obj is not None:
        p = save_trained_model(model_obj, ds_key, chosen)
        print('Saved model to', p)
    else:
        print('Failed to train chosen model.')

    return summary, chosen


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ds', default='cicids')
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--cv', type=int, default=5)
    args = p.parse_args()
    choose_model_by_cv(args.ds, sample_size=args.n, cv_folds=args.cv)
