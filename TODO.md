# FRAUD PIPELINE STABILIZATION TODO
Status: 🚀 STARTED (Approved Plan)

## Approved Edit Plan Steps

### ✅ Step 1: Create this TODO.md [DONE]

### ⏳ Step 2: Update config.py
- Standardize MODEL_CONFIG:
  * RF: n_estimators=200, max_depth=12
  * XGBoost: max_depth=7
- Remove ALL TUNING_CONFIG thresholds (keep dynamic F1-opt)
- Remove/relax aggressive TUNING_CONFIG param overrides if any

### ⏳ Step 3: Fix model_trainer.py
- CRITICAL: Add feature reindex in predict_fraud() before predict_proba()
- Remove unused CalibratedClassifierCV import

### ⏳ Step 4: Minor data_loader.py verification (logging)

### ⏳ Step 5: Test & Retrain
- execute: python offline_trainer.py
- Verify metrics: PaySim ~90-97%, CICIDS realistic, FP/FN present
- Test app.py predictions (no errors, natural probs)

### ⏳ Step 6: Mark COMPLETE + attempt_completion

**Progress: 1/6**
