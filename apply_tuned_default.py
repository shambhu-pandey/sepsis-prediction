import os, shutil, json
TUNED = 'models/best_cicids_logistic_regression_tuned_k20_c0.01.pkl'
CANON = 'models/best_cicids_logistic_regression.pkl'
BACKUP = 'models/best_cicids_logistic_regression.bak.pkl'

if os.path.exists(CANON):
    print('Backing up existing canonical model to', BACKUP)
    shutil.copy2(CANON, BACKUP)

print('Copying tuned model to canonical path...')
shutil.copy2(TUNED, CANON)

# Update evaluation_metrics.json best_model for cicids
meta_file = 'models/evaluation_metrics.json'
if os.path.exists(meta_file):
    with open(meta_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'cicids' in data:
        data['cicids']['best_model'] = 'Logistic Regression'
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print('Updated evaluation_metrics.json best_model to Logistic Regression')
    else:
        print('cicids entry not found in evaluation_metrics.json')
else:
    print('evaluation_metrics.json not found')

print('Done')

