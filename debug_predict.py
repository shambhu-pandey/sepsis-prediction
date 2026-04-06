from modules.model_trainer import load_all_models
import pandas as pd
from config import FRAUD_EXAMPLES

ds='cicids'
all_models=load_all_models(ds)
if not all_models:
    print('No models found for', ds)
    raise SystemExit(1)
name, model = next(iter(all_models.items()))
print('Using model:', name)
print('Model type:', type(model))
if hasattr(model,'named_steps'):
    print('Pipeline steps:', list(model.named_steps.keys()))
    try:
        print('Extractor raw len:', len(model.named_steps['extractor'].raw_features))
    except Exception as e:
        print('Extractor raw error', e)

print('Input example:', FRAUD_EXAMPLES.get(ds))
X=pd.DataFrame([FRAUD_EXAMPLES.get(ds)])
print('X columns:', X.columns.tolist())

try:
    p = model.predict_proba(X)
    print('predict_proba raw:', p)
    print('proba shape:', getattr(p,'shape',None))
except Exception as e:
    print('predict_proba error:', e)

try:
    preds = model.predict(X)
    print('predict:', preds)
except Exception as e:
    print('predict error:', e)

# Try manual pipeline steps
if hasattr(model,'named_steps'):
    if 'extractor' in model.named_steps:
        try:
            extracted = model.named_steps['extractor'].transform(X)
            print('extract output type/shape:', type(extracted), getattr(extracted,'shape',None))
        except Exception as e:
            print('extract error:', e)
    if 'preprocessor' in model.named_steps:
        try:
            pre = model.named_steps['preprocessor'].transform(X)
            print('preprocessor output shape:', getattr(pre,'shape',None))
        except Exception as e:
            print('preprocessor error:', e)

# print classifier attributes
try:
    clf = model.named_steps.get('classifier')
    if clf is not None:
        print('Classifier type:', type(clf))
        if hasattr(clf, 'classes_'):
            print('Classifier classes:', clf.classes_)
        if hasattr(clf, 'feature_names_in_'):
            print('Classifier feature_names_in_ len:', len(clf.feature_names_in_))
except Exception as e:
    print('classifier introspect error:', e)
