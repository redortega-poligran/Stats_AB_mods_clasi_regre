
import json, os, datetime, subprocess, sys, pathlib

ART = pathlib.Path("artifacts")
NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def get_commit_short():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "N/A"

def section_regression(reg):
    if not reg:
        return "### 2.1 Regresión\n_No hay artifacts de regresión (regression_metrics.json)._\\n"
    res = reg.get("results", {})
    lines = []
    lines.append("### 2.1 Regresión (5-Fold)")
    lines.append("")
    if res:
        lines.append("| Modelo | MAE | RMSE | R² |")
        lines.append("|---|---:|---:|---:|")
        for model, m in res.items():
            mae = m.get("MAE_mean","")
            rmse = m.get("RMSE_mean","")
            r2 = m.get("R2_mean","")
            lines.append(f"| {model} | {mae} | {rmse} | {r2} |")
    rank = reg.get("ranking_by_RMSE")
    if rank:
        best = rank[0][0]
        lines.append("")
        lines.append(f"**Mejor por RMSE:** `{best}`")
    lines.append("")
    return "\\n".join(lines) + "\\n"

def section_classification(cls):
    if not cls:
        return "### 2.2 Clasificación\n_No hay artifacts de clasificación (classif_metrics.json)._\\n"
    roc = cls.get("roc_auc", {})
    best = cls.get("best","N/A")
    lines = []
    lines.append("### 2.2 Clasificación (ROC-AUC 5-Fold)")
    lines.append("")
    if roc:
        lines.append("| Modelo | ROC-AUC |")
        lines.append("|---|---:|")
        for k,v in roc.items():
            lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    lines.append(f"**Mejor:** `{best}`. Modelo guardado en `artifacts/model_classif.pkl`.")
    lines.append("")
    return "\\n".join(lines) + "\\n"

def section_ab(ab):
    if not ab:
        return "### 2.3 A/B (simulado)\n_No hay artifacts de A/B (ab_results.json)._\\n"
    diff = ab.get("diff_abs","N/A")
    ci = ab.get("ci95",["N/A","N/A"])
    p = ab.get("p_value","N/A")
    sig = ab.get("significant_α_0.05","N/A")
    lines = []
    lines.append("### 2.3 A/B (simulado)")
    lines.append("")
    lines.append(f"- diff_abs: **{diff}**")
    lines.append(f"- IC95: **[{ci[0]}, {ci[1]}]**")
    lines.append(f"- p-valor: **{p}**")
    lines.append(f"- significativo (α=0.05): **{sig}**")
    lines.append("")
    return "\\n".join(lines) + "\\n"

def section_ts(ts):
    if not ts:
        return "### 2.4 Series de tiempo\n_No hay artifacts de series (ts_metrics.json)._\\n"
    mape = ts.get("mape","N/A")
    order = ts.get("order","N/A")
    lines = []
    lines.append("### 2.4 Series de tiempo")
    lines.append("")
    lines.append(f"- MAPE (ARIMA {order}): **{mape}%**")
    lines.append("")
    return "\\n".join(lines) + "\\n"

def build_model_report():
    reg = read_json(ART / "regression_metrics.json")
    cls = read_json(ART / "classif_metrics.json")
    ab = read_json(ART / "ab_results.json")
    ts = read_json(ART / "ts_metrics.json")

    commit = get_commit_short()
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    hdr = f"# Model Report — Proyecto (generado {NOW})\\nCommit: `{commit}` | Entorno: Python {pyver}\\n\\n"
    ctx = (
        "## 1. Contexto y objetivo\\n"
        "- Objetivo: <qué predices / por qué importa al negocio>.\\n"
        "- Datos: <fuente/dataset>, tamaño, features principales.\\n"
        "- Métricas clave: RMSE/R² (regresión), ROC-AUC (clasificación), diff/IC95/p-valor (A/B), MAPE (series).\\n\\n"
        "## 2. Resultados\\n\\n"
    )

    body = section_regression(reg) + section_classification(cls) + section_ab(ab) + section_ts(ts)

    tail = (
        "## 3. Decisiones operativas\\n"
        "- Regresión: <modelo elegido> — razón breve. Próximo: <mejora>.\\n"
        "- Clasificación: servir `<modelo>` vía API/CLI. Monitoreo: <qué mirar>.\\n"
        "- A/B: <lanzar / iterar / más muestra> por <p-valor/IC>.\\n"
        "- Series: baseline/ARIMA aceptable para <uso> con MAPE <X>.\\n\\n"
        "## 4. Riesgos y supuestos\\n"
        "- Supuestos A/B (aleatorización, independencia, mismo período).\\n"
        "- Fuga de datos mitigada con Pipeline.\\n"
        "- Sesgos potenciales / desbalance.\\n\\n"
        "## 5. Próximos pasos\\n"
        "- <mejora 1> | impacto esperado.\\n"
        "- <mejora 2> | impacto esperado.\\n"
    )

    text = hdr + ctx + body + tail
    pathlib.Path("MODEL_REPORT.md").write_text(text, encoding="utf-8")
    return text

def build_readme():
    text = (
        "# Proyecto — Ejecución rápida\\n\\n"
        "Breve: ETL + modelos (clasificación/regresión) + API + A/B sim + series. Artefactos en `artifacts/`.\\n\\n"
        "## Requisitos\\n"
        "- Python 3.11 recomendado\\n"
        "- `pip install -r requirements_day2.txt` (o fusiona con tu `requirements.txt`)\\n\\n"
        "## Cómo ejecutar (Windows PowerShell)\\n\\n"
        "### 1) Clasificación (entrena y guarda modelo)\\n"
        "```powershell\\npython -m scripts.train_classification\\n```\\n"
        "Artefactos: `artifacts/model_classif.pkl`, `artifacts/feature_names.json`, `artifacts/classif_metrics.json`\\n\\n"
        "### 2) API FastAPI\\n"
        "```powershell\\nuvicorn app.main:app --reload\\nInvoke-RestMethod http://127.0.0.1:8000/health\\nInvoke-RestMethod http://127.0.0.1:8000/schema\\n```\\n"
        "Predicción:\\n"
        "```powershell\\npython -c \\\"import json; n=json.load(open('artifacts/feature_names.json')); json.dump({k:0.0 for k in n}, open('sample.json','w'), indent=2)\\\"\\n"
        "Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -InFile .\\\\sample.json\\n```\\n\\n"
        "### 3) Simulación A/B\\n"
        "```powershell\\npython -m scripts.ab_test_sim --n 10000 --cr 0.10 --lift 0.05\\n```\\n"
        "Salida: `artifacts/ab_results.json`\\n\\n"
        "### 4) Series de tiempo (ARIMA)\\n"
        "```powershell\\npython -m scripts.ts_basics\\n```\\n"
        "Salida: `artifacts/ts_metrics.json`\\n\\n"
        "### 5) Tests\\n"
        "```powershell\\npython -m pytest -q\\n```\\n\\n"
        "## CI\\n"
        "Workflow en `.github/workflows/ci.yml`: corre tests, entrena, sim A/B, series y sube artifacts.\\n\\n"
        "## Estructura\\n"
        "````\\n"
        "scripts/\\napp/\\nartifacts/\\ntests/\\n.github/workflows/\\n"
        "````\\n"
    )
    pathlib.Path("README.md").write_text(text, encoding="utf-8")
    return text

if __name__ == "__main__":
    mr = build_model_report()
    rd = build_readme()
    print("OK -> MODEL_REPORT.md y README.md generados.")
