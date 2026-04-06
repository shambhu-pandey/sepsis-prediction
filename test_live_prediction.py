from modules.model_trainer import load_trained_model, predict_fraud
import pandas as pd

# quick smoke test using cicids default example from app
from config import FRAUD_EXAMPLES

ds = 'cicids'
models = []
# try to load best model names available
from modules.model_trainer import load_all_models
all_models = load_all_models(ds)
if not all_models:
    print('No models found for', ds)
else:
    name, model = next(iter(all_models.items()))
    print('Using model:', name)
    tx = FRAUD_EXAMPLES.get(ds, {})
    X = pd.DataFrame([tx])
    pred, prob, details = predict_fraud(model, X, name, raw_tx=tx, ds_key=ds)
    print('Prediction:', pred)
    print('Prob:', prob)
    print('Details:', details)
