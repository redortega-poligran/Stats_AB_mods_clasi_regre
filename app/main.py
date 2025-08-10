from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np

from .model_io import load_model, load_feature_names

app = FastAPI(title="Day2 Classification API", version="0.1.0")
_model = None
_features: List[str] = []

class Features(BaseModel):
    values: Dict[str, float]

@app.on_event("startup")
def _load():
    global _model, _features
    _model = load_model()
    _features = load_feature_names()

@app.get("/health")
def health():
    return {"status": "ok", "features": len(_features)}

@app.get("/schema")
def schema():
    return {"feature_names": _features}

@app.post("/predict")
def predict(payload: Features):
    missing = [f for f in _features if f not in payload.values]
    extra = [k for k in payload.values.keys() if k not in _features]
    if missing:
        raise HTTPException(status_code=400, detail={"error": "faltan features", "missing": missing})
    if extra:
        # permitimos extras pero avisamos
        pass

    x = np.array([[payload.values[f] for f in _features]], dtype=float)
    proba = float(_model.predict_proba(x)[0, 1])
    label = int(proba >= 0.5)
    return {"proba": round(proba, 6), "label": label}