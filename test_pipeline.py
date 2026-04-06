"""
Validation: Anti-Leakage Pipeline Test
Verifies that metrics are realistic (NOT 100%) after removing leakage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import prepare_dataset
from modules.model_trainer import train_and_evaluate_all, save_all_models

DATASETS = ["banksim", "paysim"]

for ds in DATASETS:
    print(f"\n{'='*60}")
    print(f"  TESTING: {ds.upper()}")
    print(f"{'='*60}")

    try:
        X_train, X_test, y_train, y_test, scaler, le, feat_names, stats = \
            prepare_dataset.__wrapped__(ds)
    except Exception as e:
        print(f"  SKIP: {e}")
        continue

    print(f"  Rows: {stats['total_rows_available']:,} -> {stats['rows_used']:,} used")
    print(f"  Features: {len(feat_names)} -> {feat_names[:6]}...")
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"  Fraud ratio: {stats['sampled_fraud_ratio']:.4%}")

    # Verify leakage cols are GONE
    leakage = ["newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest"]
    found = [c for c in leakage if c in feat_names]
    if found:
        print(f"  !! LEAKAGE DETECTED: {found}")
    else:
        print(f"  OK: No leakage columns present")

    print(f"\n  Training 4 models (with 5-fold CV)...")
    results, models, best, timings = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    print(f"\n  {'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*70}")
    for name, m in results.items():
        cv = m.get("CV-F1 (mean+/-std)", "")
        print(f"  {name:<25} {m['Accuracy']:>8.4f} {m['Precision']:>8.4f} "
              f"{m['Recall']:>8.4f} {m['F1-Score']:>8.4f} {m['ROC-AUC']:>8.4f}")

    print(f"\n  Best: {best}  |  Timings: {timings}")

    # Verify NOT perfect
    best_acc = results[best]["Accuracy"]
    if best_acc >= 0.999:
        print(f"  !! WARNING: Accuracy {best_acc} is suspiciously perfect!")
    else:
        print(f"  OK: Accuracy {best_acc} is realistic (not 100%)")

    save_all_models(models, ds)

print(f"\n{'='*60}")
print(f"  ALL TESTS COMPLETE")
print(f"{'='*60}")
