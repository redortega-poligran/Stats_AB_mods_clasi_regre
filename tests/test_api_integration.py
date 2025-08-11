# tests/test_api_integration.py
# Smoke test de la API FastAPI: asegura artifacts y valida /health, /schema y /predict.
import json
from pathlib import Path
from fastapi.testclient import TestClient

def _ensure_artifacts():
    if not Path("artifacts/model_classif.pkl").exists():
        # Entrena clasificación y guarda modelo y features
        from scripts.train_classification import main as train_main
        train_main()

_ensure_artifacts()

from app.main import app  # importa después de asegurar artifacts

def test_health_and_schema():
    # USAR CONTEXT MANAGER para que corran los eventos de startup
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"
        assert data.get("features", 0) > 0

        r2 = client.get("/schema")
        assert r2.status_code == 200
        names = r2.json().get("feature_names", [])
        assert isinstance(names, list) and len(names) > 0

def test_predict_basic_payload():
    with TestClient(app) as client:
        names = client.get("/schema").json()["feature_names"]
        payload = {"values": {n: 0.0 for n in names}}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        out = resp.json()
        assert "proba" in out and "label" in out
        assert 0.0 <= out["proba"] <= 1.0
