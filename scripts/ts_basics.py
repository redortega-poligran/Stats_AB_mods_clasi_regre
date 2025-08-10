# scripts/ts_basics.py
# Serie sintética mensual: descomposición simple y ARIMA. Guarda MAPE en artifacts/ts_metrics.json
import json, pathlib, numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)

def main():
    # Serie mensual 2015-01 .. 2024-12 (120 puntos)
    idx = pd.period_range('2015-01', '2024-12', freq='M').to_timestamp()
    n = len(idx)
    rng = np.random.default_rng(42)

    trend = np.linspace(100, 180, n)
    season = 10 * np.sin(2*np.pi*np.arange(n)/12.0)
    noise = rng.normal(0, 3, n)
    y = trend + season + noise
    s = pd.Series(y, index=idx, name='y')

    train, test = s.iloc[:-12], s.iloc[-12:]

    model = ARIMA(train, order=(1,1,1))
    res = model.fit()
    fc = res.forecast(steps=len(test))
    metric_mape = mape(test.values, fc.values)

    pathlib.Path('artifacts').mkdir(exist_ok=True, parents=True)
    out = {'mape': round(metric_mape, 3), 'n_train': len(train), 'n_test': len(test), 'order': [1,1,1]}
    with open('artifacts/ts_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(out)

if __name__ == '__main__':
    main()