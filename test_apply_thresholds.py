from modules.data_loader import prepare_dataset
from modules.model_trainer import load_trained_model, evaluate_model

Xtr,Xte,ytr,yte,_ = prepare_dataset('cicids', 5000)
model = load_trained_model('cicids','XGBoost')
met,_,_ = evaluate_model(model, Xte, yte, 'XGBoost', ds_key='cicids')
print('cicids XGBoost metrics:', met)
