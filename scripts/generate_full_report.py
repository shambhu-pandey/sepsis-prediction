import json
import pandas as pd
import os
import sys

# ensure repo root on path so local 'modules' package imports work
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from modules.model_trainer import load_all_models, predict_fraud
from config import DATASET_META, FRAUD_EXAMPLES, NORMAL_EXAMPLES

metrics_path = os.path.join(repo_root, 'models', 'evaluation_metrics.json')
out_metrics_csv = os.path.join(repo_root, 'models', 'metrics_summary.csv')
out_preds_csv = os.path.join(repo_root, 'models', 'prediction_checks.csv')

with open(metrics_path, 'r') as f:
    metrics = json.load(f)

rows = []
pred_rows = []

for ds_key, ds_meta in DATASET_META.items():
    ds_metrics = metrics.get(ds_key, {})
    results = ds_metrics.get('results', {})
    for model_name, m in results.items():
        rows.append({
            'dataset': ds_key,
            'model': model_name,
            'accuracy': m.get('Accuracy', None),
            'precision': m.get('Precision', None),
            'recall': m.get('Recall', None),
            'f1': m.get('F1-Score', None),
            'best_threshold': m.get('Best_Threshold', None)
        })

# Save metrics summary
df_metrics = pd.DataFrame(rows)
df_metrics.to_csv(out_metrics_csv, index=False)
print(f"Wrote metrics summary to {out_metrics_csv}")
print(df_metrics)

# Now run live-prediction checks using FRAUD_EXAMPLES and NORMAL_EXAMPLES
for ds_key in DATASET_META:
    models = load_all_models(ds_key)
    fraud_ex = FRAUD_EXAMPLES.get(ds_key, {})
    normal_ex = NORMAL_EXAMPLES.get(ds_key, {})
    for model_name, model in models.items():
        for ex_type, ex in (('fraud', fraud_ex), ('normal', normal_ex)):
            if not ex:
                continue
            X = pd.DataFrame([ex])
            pred, prob, details = predict_fraud(model, X, model_name, raw_tx=ex, ds_key=ds_key)
            pred_rows.append({
                'dataset': ds_key,
                'model': model_name,
                'example_type': ex_type,
                'pred': int(pred),
                'probability': float(prob),
                'threshold_used': float(details.get('threshold', None)),
                'risk_level': details.get('risk_level')
            })

if pred_rows:
    df_preds = pd.DataFrame(pred_rows)
    df_preds.to_csv(out_preds_csv, index=False)
    print(f"Wrote prediction checks to {out_preds_csv}")
    print(df_preds)
else:
    print('No models or examples found to run prediction checks.')
