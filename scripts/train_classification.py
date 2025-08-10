# scripts/train_classification.py
# Compara Dummy vs LogReg vs RandomForest (ROC-AUC con 5-fold CV) y guarda el mejor modelo.
import json, pathlib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def eval_model(model, X, y):
    cv = cross_validate(model, X, y, cv=CV, scoring='roc_auc', n_jobs=-1, return_estimator=False)
    return float(np.mean(cv['test_score']))

def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    results = {}
    results['Dummy'] = eval_model(DummyClassifier(strategy='most_frequent'), X, y)
    logreg = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200, solver='liblinear'))
    results['LogisticRegression'] = eval_model(logreg, X, y)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    results['RandomForest'] = eval_model(rf, X, y)

    # Elige el mejor por ROC-AUC
    best_name = max(results.items(), key=lambda kv: kv[1])[0]
    if best_name == 'LogisticRegression':
        best_model = logreg.fit(X, y)
    else:
        best_model = rf.fit(X, y)

    pathlib.Path('artifacts').mkdir(exist_ok=True, parents=True)
    joblib.dump(best_model, 'artifacts/model_classif.pkl')
    with open('artifacts/feature_names.json', 'w') as f:
        json.dump(list(X.columns), f, indent=2)

    out = {'roc_auc': results, 'best': best_name}
    print(json.dumps(out, indent=2))
    with open('artifacts/classif_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()