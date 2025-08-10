# Test rápido: clasificación con ROC-AUC alto en dataset breast_cancer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def test_classification_roc_auc():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200, solver='liblinear'))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    assert scores.mean() > 0.95