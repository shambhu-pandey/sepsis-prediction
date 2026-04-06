# Fraud Detection Project — Simple Step-by-Step Guide

This guide explains this fraud detection project in very simple words. It covers everything from raw data to running the app. Read it slowly and try the code examples.

---

## 1. INTRODUCTION

### What is fraud detection?
- Fraud detection is finding bad or fake transactions.
- A transaction is money moving from one user to another.
- Fraud means someone tries to steal money or act wrongly.

### Why fraud detection is important
- It protects people's money.
- It keeps companies from losing money.
- It helps users trust the service.

### Real-life examples
- Bank fraud: someone takes money from an account without permission.
- Credit card fraud: stolen card used to buy things.
- Fake transfers: moving money to fake accounts.

---

## 2. DATASET EXPLANATION

This project uses several public datasets. Each dataset has rows (transactions) and columns (features).

### PaySim
- Synthetic (simulated) mobile money transactions.
- Key columns: `type` (TRANSFER, CASH_OUT...), `amount`, `oldbalanceOrg`, `oldbalanceDest`, and `isFraud` (target).
- `isFraud`: 1 means fraud, 0 means not fraud.

### BankSim
- Another simulated bank transaction dataset.
- Columns similar to PaySim (amounts, account types, timestamps).
- Has an `isFraud` or `fraud` column as target.

### CICIDS2017
- Network traffic dataset for intrusion (abnormal traffic).
- Each row is a network connection, not a money transaction.
- Target column may be `Label` which indicates attack types.

### IEEE-CIS
- Real-world credit card transaction dataset (often from competitions).
- Many features about the card, device and transaction.
- Target column is often `isFraud` or `Class`.

What the target column means: `isFraud` or `Class` tells us the correct answer during training. We teach the model to predict this value.

---

## 3. DATA LOADING

We use `pandas` to read CSV files. It is simple.

Example code:

```python
import pandas as pd
# load the PaySim cleaned file
df = pd.read_csv('data/paysim_clean.csv')
print(df.shape)       # shows rows, columns
print(df.columns)     # shows column names
print(df.head())      # shows first 5 rows
```

Explain columns and structure:
- Each row is one transaction.
- Columns store information like amount, type, and balances.
- We always keep the target (e.g., `isFraud`) separate from inputs.

---

## 4. DATA PREPROCESSING

We must clean and prepare data before training. This is called preprocessing.

### Handling missing values
- Some cells may be empty (missing).
- Simple fix: fill numbers with 0 or the column mean.

Example:
```python
# fill missing numbers with 0
num_cols = df.select_dtypes(include=['float','int']).columns
df[num_cols] = df[num_cols].fillna(0)
# fill missing strings with 'unknown'
df = df.fillna('unknown')
```

### Encoding categorical data
- Many columns are words (like `type`). Models need numbers.
- `OneHotEncoder` turns each category into a column of 0/1.

Example (simple with pandas):
```python
# turn type into columns: type_CASH_OUT, type_TRANSFER, ...
df = pd.get_dummies(df, columns=['type'])
```

Or using sklearn pipeline:
```python
from sklearn.preprocessing import OneHotEncoder
# OneHotEncoder is used inside a ColumnTransformer in the project
```

### Feature selection
- Remove the target column from inputs before training.
- Remove leakage columns (columns that show the answer).

Example:
```python
X = df.drop(columns=['isFraud'])
y = df['isFraud']
```

### Why data leakage is dangerous
- Data leakage: the model sees answers during training.
- If leakage exists, the model looks perfect but will fail in real life.
- Always drop columns that directly tell if the row is fraud.

---

## 5. DATA LEAKAGE (VERY IMPORTANT)

### What is data leakage?
- When training data contains secret info that should not be available at prediction time.
- Example: if the dataset has a `label` column and you accidentally keep it as a feature.

### Why 100% accuracy is wrong
- If a model uses leaked info, it can give 100% accuracy on test data, but it won't generalize.
- Real-world performance will be poor.

### Wrong vs correct code example
Wrong (leak):
```python
# BAD: Keeping target inside X
X = df  # contains isFraud
y = df['isFraud']
model.fit(X, y)  # model can cheat
```

Correct:
```python
X = df.drop(columns=['isFraud'])
y = df['isFraud']
model.fit(X, y)
```

Always check the column names and drop obvious leak columns: `isFraud`, `fraud`, `label`, `Class`.

---

## 6. CLASS IMBALANCE

### What is imbalance?
- A dataset is imbalanced when one class has many more samples than the other.
- In fraud data, fraud rows are rare (like 0.1% of transactions).

### Why fraud datasets are imbalanced
- Most transactions are normal.
- Fraud is rare but important.

### SMOTE vs SMOTETomek (simple)
- SMOTE: creates new synthetic fraud samples to balance classes.
- SMOTETomek: first adds synthetic frauds (SMOTE), then removes some noisy samples (Tomek links). It cleans the border between classes.

### Why we used SMOTETomek
- Gives a better balance and cleaner data for the model to learn.
- Helps avoid creating noisy synthetic examples that confuse the model.

Small code example (project uses imbalanced-learn):
```python
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X, y)
```

Note: Resampling is done inside the training pipeline only (not on test set).

---

## 7. MACHINE LEARNING MODELS

We use simple models that work well: Logistic Regression, Random Forest, XGBoost.

### Logistic Regression
- What: A linear model that predicts probability with a formula.
- When to use: quick, small data, easy to explain.
- Why used: fast and interpretable baseline.

### Random Forest
- What: Many decision trees combined; each tree votes.
- When to use: works well on tabular data, handles non-linear patterns.
- Why used: good performance, robust to different data types.

### XGBoost
- What: Gradient boosted trees — trees built in sequence to fix errors.
- When to use: high performance on tabular data, often top in competitions.
- Why used: strong performance, handles complex patterns.

---

## 8. MODEL TRAINING

### Train-test split
- We split data into training and test sets.
- Training set: model learns from this.
- Test set: model is evaluated on unseen data.

Example:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Why split data
- To check if the model really learned patterns, not memorized the training data.

### Pipeline concept (simple)
- A pipeline chains preprocessing steps and the model together.
- It keeps the same steps for training and live prediction.

Example (very small):
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression())])
pipe.fit(X_train, y_train)
```

In this project, pipelines include feature extraction, encoding, sampling (SMOTETomek), and classifier.

---

## 9. EVALUATION METRICS

We use these to judge models.

### Accuracy
- Percent of correct predictions.
- Not good alone when data is imbalanced.
- Example: If fraud is 1% and model always predicts not-fraud, accuracy = 99% but useless.

### Precision
- Of the transactions predicted as fraud, how many are actually fraud?
- High precision means few false alarms.

### Recall (also sensitivity)
- Of all real frauds, how many did we catch?
- High recall means we miss few frauds.

### F1-score
- Balance between precision and recall. It is the harmonic mean.

### Confusion Matrix
- Table of counts:
  - True Positive (TP): predicted fraud and truly fraud.
  - False Positive (FP): predicted fraud but actually normal.
  - True Negative (TN): predicted normal and truly normal.
  - False Negative (FN): missed fraud.

Real-life consequence: If recall is low, many frauds slip through and money is lost. If precision is low, many normal users get blocked.

---

## 10. THRESHOLD LOGIC

### What is probability?
- Many models return a probability (e.g., 0.28) that a transaction is fraud.

### What is threshold?
- A number (like 0.25). If the probability >= threshold, we flag it as fraud.

### Why 0.5 is not always good
- Default of 0.5 may be wrong when classes are imbalanced.
- Better to pick threshold that balances precision and recall for business needs.

### How threshold affects detection
- Lower threshold → more blocks (higher recall, lower precision).
- Higher threshold → fewer blocks (lower recall, higher precision).

Example:
```python
prob = model.predict_proba(X)[0,1]
threshold = 0.25
is_fraud = int(prob >= threshold)
```

---

## 11. LIVE PREDICTION

### How input is taken
- The app collects transaction fields (amount, type, balances) via a form.

### How model predicts
- Input is turned into a DataFrame with the same columns the pipeline expects.
- The pipeline runs preprocessing and then `predict_proba` or `predict`.

### What is `predict_proba()`?
- Method on many classifiers that returns probability for each class.
- Example: `model.predict_proba(X)[0,1]` is probability of fraud for the first row.

### Why feature alignment is important
- The model expects columns in the same order/names as during training.
- If columns are missing or in different order, prediction can be wrong or crash.
- The project uses an extractor to ensure alignment before predicting.

Simple code for predicting a single transaction:
```python
import pandas as pd
# tx is a dict with input values
X_new = pd.DataFrame([tx])
prob = pipe.predict_proba(X_new)[0,1]
```

---

## 12. COMMON ERRORS (VERY IMPORTANT)

### 100% accuracy problem
- Usually caused by data leakage. Fix by removing leaked columns.

### ML probability = 0 problem
- Sometimes `predict_proba` returns 0 if model or data mismatch.
- Fix: ensure `predict_proba` exists, or use `decision_function` converted with sigmoid.

### Feature mismatch error
- Happens when input columns do not match expected features.
- Fix: reindex to expected features or fill missing features with zeros.

### Overfitting
- Model learns training data too well, fails on new data.
- Fixes: use simpler models, more data, regularization, cross-validation.

---

## 13. FINAL SYSTEM ARCHITECTURE

Simple flow:

Dataset → Preprocessing → Model → Prediction → Output

- Dataset: CSV files in `data/`
- Preprocessing: cleaning, encoding, scaling, resampling
- Model: trained classifier inside `models/`
- Prediction: pipeline applied to new transaction
- Output: risk label and probability shown in app

---

## 14. VIVA QUESTIONS & ANSWERS

Q: What is fraud detection?
A: Finding and stopping bad transactions.

Q: Why accuracy is not enough?
A: Because class imbalance can hide failures (e.g., always predicting normal gives high accuracy).

Q: What is data leakage?
A: When the model sees the answer during training (makes it cheat).

Q: Why SMOTETomek used?
A: To balance classes and clean noisy examples for better learning.

Q: Difference between Random Forest & XGBoost?
A: Random Forest builds many trees independently and averages them. XGBoost builds trees one after another to fix errors (boosting). XGBoost often gets better results but needs tuning.

Q: Why threshold is important?
A: It decides when to flag a transaction as fraud; it controls false alarms vs missed fraud.

---

## 15. SIMPLE CODE WALKTHROUGH

Files to look at:
- [modules/data_loader.py](modules/data_loader.py)
- [modules/model_trainer.py](modules/model_trainer.py)

### `data_loader.py` (simple)
- Responsible for reading CSV and extracting features.
- It may contain a `DomainFeatureExtractor` that maps raw fields to pipeline inputs.

### `model_trainer.py` (core)
- Trains models and builds pipelines.
- Key functions:
  - `build_model_pipeline(...)` — builds a pipeline of preprocessing and classifier.
  - `get_probabilities(model, X, name)` — returns probability of fraud for rows.
  - `predict_fraud(model, X_input, model_name, raw_tx=None, ds_key=None)` — used by the app for a single transaction. It aligns features, computes proba, chooses threshold, and returns risk.

Example usage of `predict_fraud()`:
```python
from modules.model_trainer import predict_fraud
pred, prob, details = predict_fraud(best_model, X_new, 'Random Forest', raw_tx=tx, ds_key='paysim')
```

---

## 16. SUMMARY

What you learned:
- How fraud detection works step-by-step.
- Why data cleaning, avoiding leakage, and handling imbalance matter.
- How pipelines and thresholds shape predictions.

Why the project is useful:
- Shows a realistic way to detect fraud with simple tools.
- Teaches how to build, evaluate, and deploy a model safely.

Real-world application:
- Banks or payment apps can use this to flag suspicious transactions and protect users.

---

## Quick Run Tips

To run the app locally (from project root):
```bash
# activate your virtualenv
.venv\Scripts\Activate.ps1  # on Windows PowerShell
# then
streamlit run app.py
```

Open the browser link Streamlit shows.

---

## Final notes for exam / review
- Explain simply: always say why you drop `isFraud` from inputs.
- Show one example of how threshold changes behavior (try 0.05 vs 0.5).
- Remember: never train and test on the same data without proper split.

Good luck with your review — read this file slowly and try the example code.

