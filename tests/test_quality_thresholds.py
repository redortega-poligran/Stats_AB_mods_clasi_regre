# tests/test_quality_thresholds.py
import json, pathlib

def test_auc_minimo_clasificacion():
    p = pathlib.Path("artifacts/classif_metrics.json")
    data = json.loads(p.read_text())
    best_auc = max(data["roc_auc"].values())
    assert best_auc > 0.95

def test_mape_max_series():
    p = pathlib.Path("artifacts/ts_metrics.json")
    data = json.loads(p.read_text())
    assert data["mape"] < 20.0
