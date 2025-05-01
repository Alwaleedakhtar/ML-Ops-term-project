
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

warnings.filterwarnings('ignore')


# SETUP: MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow server must be running locally


# Load Dataset
df = pd.read_csv("dataset_.csv")
df.columns = [
    'age', 'income', 'home_ownership', 'emp_length', 'loan_purpose',
    'loan_grade', 'loan_amount', 'interest_rate', 'loan_status',
    'percent_income', 'defaulted', 'credit_history_length'
]

# Clean missing values
df = df[(df['emp_length'] != '?') & (df['interest_rate'] != '?')]
df['emp_length'] = df['emp_length'].astype(int)
df['interest_rate'] = df['interest_rate'].astype(float)
df['label'] = df['defaulted'].map({'Y': 1, 'N': 0})
df.drop(columns=['defaulted'], inplace=True)

# Encode categoricals
categorical_cols = ['home_ownership', 'loan_purpose', 'loan_grade', 'loan_status']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Train/test split
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# BASELINE MODEL
with mlflow.start_run(run_name="baseline_logistic_regression"):
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    })
    mlflow.sklearn.log_model(model, "model")
    print(classification_report(y_test, y_pred))


# HYPEROPT TUNING
def objective(params):
    with mlflow.start_run(nested=True):
        params['max_iter'] = int(params['max_iter'])
        model = LogisticRegression(C=params['C'], max_iter=params['max_iter'], solver=params['solver'],
                                   class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        print(f"\nParams: {params} | F1: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        return {'loss': -f1, 'status': STATUS_OK}

search_space = {
    'C': hp.loguniform('C', -4, 2),
    'max_iter': hp.quniform('max_iter', 100, 1000, 50),
    'solver': hp.choice('solver', ['liblinear', 'lbfgs', 'saga'])
}

with mlflow.start_run(run_name="logreg_hyperopt_tuning"):
    trials = Trials()
    best_result = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=20, trials=trials)
    print("Best hyperparameters found:", best_result)


# FINAL MODEL (DEPLOYED)
best_params = {'C': 0.0449, 'max_iter': 850, 'solver': 'liblinear'}
final_model = LogisticRegression(C=best_params['C'], max_iter=best_params['max_iter'],
                                 solver=best_params['solver'], class_weight='balanced', random_state=42)
final_model.fit(X_train, y_train)

with mlflow.start_run(run_name="final_deployment_model"):
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(final_model, "model", registered_model_name="LoanDefaultModel")
    print(classification_report(y_test, final_model.predict(X_test)))


# SIMULATE PREDICTION & DRIFT
def predict_loan_default(new_data_df):
    pred = final_model.predict(new_data_df)
    proba = final_model.predict_proba(new_data_df)
    return pred, proba

sample = X_test.iloc[[0]]
pred, prob = predict_loan_default(sample)
print("\nPrediction:", "Default" if pred[0] == 1 else "No Default")
print("Probability of Default:", prob[0][1])

original_f1 = f1_score(y_test, final_model.predict(X_test))
X_test_drifted = X_test.copy()
X_test_drifted['age'] += np.random.randint(5, 15, size=X_test_drifted.shape[0])
X_test_drifted['income'] *= np.random.uniform(0.5, 1.5, size=X_test_drifted.shape[0])
drifted_f1 = f1_score(y_test, final_model.predict(X_test_drifted))
print(f"\nOriginal F1: {original_f1:.4f} | Drifted F1: {drifted_f1:.4f}")
