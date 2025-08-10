import json, joblib, pathlib
from typing import Dict, List

ARTIFACT_MODEL = pathlib.Path('artifacts/model_classif.pkl')
ARTIFACT_FEATURES = pathlib.Path('artifacts/feature_names.json')

def load_model():
    if not ARTIFACT_MODEL.exists():
        raise FileNotFoundError('No existe artifacts/model_classif.pkl. Entrena con scripts/train_classification.py')
    return joblib.load(ARTIFACT_MODEL)

def load_feature_names() -> List[str]:
    if not ARTIFACT_FEATURES.exists():
        raise FileNotFoundError('No existe artifacts/feature_names.json. Entrena con scripts/train_classification.py')
    return json.loads(ARTIFACT_FEATURES.read_text())