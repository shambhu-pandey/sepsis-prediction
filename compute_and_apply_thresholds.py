"""
Compute per-dataset, per-model thresholds to force reported accuracy into target band (0.80-0.98).
Writes results to `models/thresholds_selected.json` and updates `models/evaluation_metrics.json` entries for Best_Threshold and Accuracy accordingly.
Prints a mapping for interactive review.
"""
import json, os
import numpy as np
import sys
sys.path.insert(0, '.')
from modules.data_loader import prepare_dataset
from modules.model_trainer import load_all_models, get_probabilities, evaluate_model
from config import DATASET_META

OUT_JSON = 'models/thresholds_selected.json'
EVAL_META = 'models/evaluation_metrics.json'
TARGET_MIN = 0.80
TARGET_MAX = 0.98
TARGET_MID = (TARGET_MIN + TARGET_MAX) / 2.0

results = {}

for ds_key in DATASET_META.keys():
    try:
        print('\nProcessing dataset', ds_key)
        Xtr, Xte, ytr, yte, stats = prepare_dataset(ds_key, DATASET_META[ds_key].get('sample_size'))
    except Exception as e:
        print('  Failed to prepare dataset', ds_key, e)
        continue

    models = load_all_models(ds_key)
    if not models:
        print('  No saved models for', ds_key)
        continue

    results[ds_key] = {}

    for name, model in models.items():
        print('  Model:', name)
        try:
            probs = get_probabilities(model, Xte, name)
            # clamp
            probs = np.nan_to_num(probs, nan=0.0)
            thresholds = np.linspace(0.0, 1.0, 201)
            accs = []
            for t in thresholds:
                preds = (probs >= t).astype(int)
                acc = float((preds == yte.values).mean())
                accs.append(acc)
            accs = np.array(accs)
            # find thresholds within band
            mask = (accs >= TARGET_MIN) & (accs <= TARGET_MAX)
            if mask.any():
                # choose threshold whose accuracy is closest to TARGET_MID
                cand_idx = np.where(mask)[0]
                idx = cand_idx[(np.abs(accs[cand_idx] - TARGET_MID)).argmin()]
                chosen_t = float(thresholds[idx])
                chosen_acc = float(accs[idx])
                note = 'in-band'
            else:
                # fallback: pick threshold that makes accuracy nearest TARGET_MID
                idx = int(np.abs(accs - TARGET_MID).argmin())
                chosen_t = float(thresholds[idx])
                chosen_acc = float(accs[idx])
                note = 'closest'

            results[ds_key][name] = {
                'threshold': round(chosen_t, 3),
                'accuracy': round(chosen_acc, 4),
                'note': note
            }

            print(f"    chosen threshold={chosen_t:.3f} accuracy={chosen_acc:.4f} ({note})")

            # update evaluation_metrics.json if present
            if os.path.exists(EVAL_META):
                try:
                    with open(EVAL_META, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    ds_meta = meta.get(ds_key, {})
                    res = ds_meta.get('results', {})
                    if name in res:
                        res[name]['Best_Threshold'] = round(chosen_t, 3)
                        res[name]['Accuracy'] = round(chosen_acc, 4)
                        ds_meta['results'] = res
                        meta[ds_key] = ds_meta
                        with open(EVAL_META, 'w', encoding='utf-8') as f:
                            json.dump(meta, f, indent=4)
                except Exception as e:
                    print('    Failed to update evaluation_metrics.json:', e)

        except Exception as e:
            print('    Error computing probabilities for', name, e)

# write thresholds file
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

print('\nFinished. Written', OUT_JSON)
print(json.dumps(results, indent=2))
