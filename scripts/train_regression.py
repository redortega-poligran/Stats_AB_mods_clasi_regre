# scripts/train_regression.py
# Compara Dummy vs LinearRegression vs RandomForest con 5-fold CV en un dataset LOCAL (no necesita internet).
# Salida: artifacts/regression_metrics.json con métricas promedio (MAE, RMSE, R2) y ranking por RMSE.

import json, pathlib, numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def eval_model(model, X, y):
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv = cross_validate(model, X, y, cv=CV, scoring=scoring, n_jobs=-1)
    return {
        "MAE_mean": round(-cv["test_mae"].mean(), 6),
        "RMSE_mean": round(-cv["test_rmse"].mean(), 6),
        "R2_mean": round(cv["test_r2"].mean(), 6),
    }

def main():
    data = load_diabetes()  # dataset pequeño incluido en sklearn (regresión)
    X, y = data.data, data.target

    results = {}
    results["DummyRegressor"] = eval_model(DummyRegressor(strategy="mean"), X, y)

    lin = make_pipeline(StandardScaler(), LinearRegression())
    results["LinearRegression"] = eval_model(lin, X, y)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
    )
    results["RandomForest"] = eval_model(rf, X, y)

    ranking = sorted(results.items(), key=lambda kv: kv[1]["RMSE_mean"])
    out = {"results": results, "ranking_by_RMSE": ranking}
    print(json.dumps(out, indent=2))

    pathlib.Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open("artifacts/regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
