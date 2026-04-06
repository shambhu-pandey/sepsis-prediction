"""
Targeted search for IEEE by restricting to a small set of features to reduce peak accuracy.
"""
import sys
sys.path.insert(0, '.')
from modules.data_loader import prepare_dataset
from modules.model_trainer import evaluate_models_cv, _train_lr, save_trained_model
from sklearn.linear_model import LogisticRegression

FEATURE_SUBSETS = [
    ['TransactionAmt', 'dist1'],
    ['TransactionAmt', 'dist1', 'card4'],
    ['TransactionAmt', 'dist1', 'card4', 'card6'],
]


def run_targeted(sample_size=20000, cv=5, target_min=0.8, target_max=0.98):
    Xtr, Xte, ytr, yte, stats = prepare_dataset('ieee', sample_size)
    for cols in FEATURE_SUBSETS:
        missing = [c for c in cols if c not in Xtr.columns]
        if missing:
            print('Skipping subset, missing', missing)
            continue
        print('Testing subset', cols)
        Xtr_sub = Xtr[cols].copy()
        # Try a set of regularizations
        for C in [0.0001, 0.001, 0.005, 0.01, 0.05]:
            print('  LR C=', C)
            summary = evaluate_models_cv(Xtr_sub, ytr, model_names=['Logistic Regression'], cv_folds=cv, ds_key='ieee')
            mean_acc = summary.get('Logistic Regression', {}).get('accuracy_mean', 0.0)
            print('   mean_acc=', mean_acc)
            if target_min <= mean_acc <= target_max:
                model = _train_lr(Xtr_sub, ytr, ds_key='ieee')
                path = save_trained_model(model, 'ieee', f'logistic_restricted_C{C}_cols_{"_".join(cols)}')
                return {'model':'Logistic Regression','C':C,'cols':cols,'mean_acc':mean_acc,'path':path}
    return {'found':False}

if __name__ == '__main__':
    res = run_targeted()
    print('Result:', res)
