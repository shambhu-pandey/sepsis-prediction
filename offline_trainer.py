import os
import json
import joblib
import pandas as pd
import warnings
from modules.data_loader import prepare_dataset
from modules.model_trainer import train_and_evaluate_all, save_all_models, MODELS_DIR
from config import DATASET_META

warnings.filterwarnings("ignore")

def _validate_results(all_metrics):
    """
    Post-training validation:
      - No dataset shows constant 1.0 accuracy across ALL models
      - Best model's confusion matrix has FP > 0 and FN > 0
      - Metrics differ across datasets
    """
    print("\n--- VALIDATION ---")
    issues = []

    accuracies_by_ds = {}

    for ds_key, ds_data in all_metrics.items():
        results = ds_data.get("results", {})
        if not results:
            issues.append(f"  [WARN] {ds_key}: No model results found")
            continue

        # Check 1: Not all models should be 1.0 accuracy
        accs = [v.get("Accuracy", 0) for v in results.values()]
        if accs and all(a == 1.0 for a in accs):
            issues.append(f"  [FAIL] {ds_key}: ALL models show 1.0 accuracy — likely data leakage!")
        else:
            print(f"  [PASS] {ds_key}: Accuracy varies across models ({min(accs):.4f} - {max(accs):.4f})")

        accuracies_by_ds[ds_key] = accs

        # Check 2: Best model should have FP > 0 and FN > 0
        best = ds_data.get("best_model")
        if best and best in results:
            cm = results[best].get("CM", {})
            fp = cm.get("FP", 0)
            fn = cm.get("FN", 0)
            if fp == 0 and fn == 0:
                issues.append(f"  [WARN] {ds_key}/{best}: CM has FP=0 AND FN=0 — suspiciously perfect")
            else:
                print(f"  [PASS] {ds_key}/{best}: CM has FP={fp}, FN={fn} (realistic)")

    # Check 3: Metrics should differ across datasets
    if len(accuracies_by_ds) > 1:
        avg_accs = {k: sum(v)/len(v) for k, v in accuracies_by_ds.items() if v}
        vals = list(avg_accs.values())
        if len(set(round(v, 4) for v in vals)) == 1:
            issues.append("  [WARN] All datasets have identical average accuracy — suspicious")
        else:
            print(f"  [PASS] Metrics differ across datasets: {avg_accs}")

    if issues:
        print("\n  ⚠️  VALIDATION ISSUES:")
        for issue in issues:
            print(issue)
    else:
        print("\n  ✅ All validation checks passed!")

    return len(issues) == 0


def run_offline_training():
    print("================================================================")
    print(" INITIALIZING OFFLINE MODEL TRAINING")
    print("================================================================\n")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    all_metrics = {}
    master_stats = {}

    for ds_key, meta in DATASET_META.items():
        print(f"[*] Processing {meta['short_name']}...")
        print("  -> Loading and Preprocessing dataset...")
        
        try:
            Xtr, Xte, ytr, yte, stats = prepare_dataset(ds_key)
        except Exception as e:
            print(f"  [CRITICAL ERROR] Failed to prepare {ds_key}: {e}")
            import traceback
            traceback.print_exc()
            raise e

        print(f"  -> Training features: {list(Xtr.columns)}")
        print(f"  -> Train shape: {Xtr.shape}, Fraud ratio: {ytr.mean():.4f}")
        print("  -> Training models...")
        results, models, best, timings = train_and_evaluate_all(Xtr, ytr, Xte, yte, ds_key=ds_key)
        
        # Save models
        save_all_models(models, ds_key)
        
        all_metrics[ds_key] = {
            "best_model": best,
            "results": results,
            "timings": timings,
            "feature_names": Xtr.columns.tolist(),
        }
        master_stats[ds_key] = stats
        
        # Print per-model results
        for mname, mresult in results.items():
            marker = " ◀ BEST" if mname == best else ""
            print(f"    {mname}: Acc={mresult.get('Accuracy',0):.4f} "
                  f"F1={mresult.get('F1-Score',0):.4f} "
                  f"AUC={mresult.get('ROC-AUC',0):.4f} "
                  f"Thresh={mresult.get('Best_Threshold',0):.3f}{marker}")
        print()

    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    stats_path = os.path.join(MODELS_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(master_stats, f, indent=4)

    # Run validation
    _validate_results(all_metrics)

    print("\n================================================================")
    print(" TRAINING COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    run_offline_training()
