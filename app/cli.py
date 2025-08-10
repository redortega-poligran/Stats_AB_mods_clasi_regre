# CLI: etl-predict --json sample.json
import argparse, json
from .model_io import load_model, load_feature_names
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='ruta a JSON con dict feature->valor')
    args = parser.parse_args()

    model = load_model()
    features = load_feature_names()
    vals = json.loads(open(args.json).read())

    x = np.array([[vals[f] for f in features]], dtype=float)
    proba = float(model.predict_proba(x)[0,1])
    print(json.dumps({'proba': round(proba, 6), 'label': int(proba >= 0.5)}, indent=2))

if __name__ == '__main__':
    main()