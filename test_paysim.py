import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import prepare_dataset
from modules.model_trainer import train_and_evaluate_all

print("Testing paysim...")
X_tr, X_te, y_tr, y_te, sc, cat_cols, fn, stats = prepare_dataset("paysim")

print(f"Features: {fn}")
r, m, best, t = train_and_evaluate_all(X_tr, y_tr, X_te, y_te)

print("Best model:", best)
for name, met in r.items():
    print(f"  {name}: {met}")

print("Done")
