# Arquitectura mínima — Stats_AB_mods_clasi_regre (v0.1)

**Objetivo**: entrenar modelos (clasificación y regresión), simular A/B y ejercicios de series; exponer un modelo vía API/CLI; automatizar con CI para dejar *artefactos* auditables.

**Alcance del repo**: *standalone*. No depende del repo del Día 1. Usa datasets de `scikit-learn` y datos sintéticos para A/B y series.

---

## 1) Vista lógica (qué piezas hay)
- **Fuentes**:
  - Clasificación: `sklearn.datasets.load_breast_cancer` (local, sin internet).
  - Regresión: `sklearn.datasets.load_diabetes`.
  - A/B: datos **sintéticos** (binomial) según `n`, `cr`, `lift`.
  - Series: **sintética** mensual con tendencia + estacionalidad.
- **Procesamiento**:
  - `scripts/train_classification.py` (5-fold ROC-AUC, guarda mejor modelo `.pkl`).
  - `scripts/train_regression.py` (5-fold MAE/RMSE/R², ranking por RMSE).
  - `scripts/ab_test_sim.py` (diff, IC95, p-valor).
  - `scripts/ts_basics.py` (ARIMA(1,1,1), MAPE).
- **Almacenamiento**:
  - Carpeta **`artifacts/`** con:
    - `model_classif.pkl`, `feature_names.json`
    - `classif_metrics.json`, `regression_metrics.json`
    - `ab_results.json`, `ts_metrics.json`
- **Serving**:
  - **API FastAPI** en `app/main.py` → `/health`, `/schema`, `/predict`.
  - **CLI** en `app/cli.py` → `etl-predict --json sample.json`.
- **Calidad**:
  - `pytest` unitario y de integración (`tests/test_api_integration.py`).
  - (Opcional) límites de calidad: AUC mínimo / MAPE máximo.
- **Automatización**:
  - **GitHub Actions** `.github/workflows/ci.yml`:
    1) instala deps
    2) **entrena** (clasif/reg/AB/series)
    3) corre **pytest**
    4) publica **artifacts** con el SHA del commit

---

## 2) Diagrama (alto nivel)

```
[Datasets sklearn / Sintéticos]
             |
             v
    [Entrenamiento & Métricas]
     |   |        |        |
     |   |        |        '--> (Series)  ->  ts_metrics.json
     |   |        '------------> (A/B sim) ->  ab_results.json
     |   '------------------------> (Regresión) -> regression_metrics.json
     '----------------------------> (Clasificación) -> classif_metrics.json + model_classif.pkl
                                      |
                                      v
                         [API FastAPI / CLI predict]
                                      |
                                      v
                               [Cliente/Consumidor]

                      [CI (pytest + training + artifacts)]
```

---

## 3) Contratos y artefactos

**API**  
- `GET /health` → `{ "status":"ok", "features": <int> }`  
- `GET /schema` → `{ "feature_names": [ ... ] }`  
- `POST /predict` → body: `{ "values": { "<feature>": <float>, ... } }`  
  Respuesta: `{ "proba": <0..1>, "label": 0|1 }`

**Artefactos (salida del pipeline)**  
- `artifacts/classif_metrics.json` → AUC por modelo + `best`  
- `artifacts/model_classif.pkl` + `feature_names.json`  
- `artifacts/regression_metrics.json` → MAE/RMSE/R² (5-fold)  
- `artifacts/ab_results.json` → diff, IC95, p-valor, significativo α=0.05  
- `artifacts/ts_metrics.json` → MAPE, orden ARIMA

---

## 4) Decisiones (ADR corto)

**[ADR-001] Artifacts en filesystem**  
- *Por qué*: rápido de generar, versionable, fácil de inspeccionar y publicar en CI.  
- *Trade-offs*: no es un registry; para producción usar MLflow/SageMaker/Vertex.  
- *Evolución*: nombrar con timestamp/commit (`model_classif_YYYYMMDD_SHA.pkl`).

**[ADR-002] FastAPI para servir el modelo**  
- *Por qué*: validación de payload con Pydantic y `/docs` auto.  
- *Alternativas*: Flask (mínimo), gRPC (contratos binarios).  
- *Evolución*: contenedor + despliegue gestionado + monitoreo (latencia y tasa de error).

**[ADR-003] Orden de CI: entrenar → tests → publicar**  
- *Por qué*: el test de integración necesita el modelo cargado.  
- *Trade-offs*: mayor tiempo de CI; mitigable con cache.  
- *Evolución*: separar jobs (train sólo en main, tests en PR).

---

## 5) Riesgos y guardas
- Overfitting si cambian features → el esquema vive en `feature_names.json`.  
- Rendimiento inestable → fija umbrales (AUC mínimo, MAPE máximo).  
- Reproducibilidad → `requirements*.txt` y versiones pinneadas.

---

## 6) Próximos pasos
- Agregar `tests/test_quality_thresholds.py` (umbrales).  
- Versionado de artifacts por fecha/SHA.  
- (Opcional) Integrar como consumidor de un ETL externo (otro repo) leyendo un CSV/Parquet y repitiendo el flujo.
