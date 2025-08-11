# QUIZ_RESPUESTAS

## 1. ¿Por qué necesitas un baseline antes de comparar modelos?
Porque sin un punto de referencia (p. ej., predecir la media), no sabes si tu modelo aporta valor. Si no supera al baseline, no sirve.

## 2. ¿Cuándo preferirías RMSE sobre MAE? Da un ejemplo.
Cuando los errores grandes te duelen mucho más que los pequeños (RMSE los penaliza más). Ej.: predicción de tiempo de entrega: llegar 2 horas tarde debe pesar mucho más que 10 min.

## 3. Explica con tus palabras qué es data leakage y cómo lo evita Pipeline.
Leakage es cuando info del conjunto de validación se cuela en el entrenamiento (ej., escalado con estadísticas de todo el dataset). Pipeline aplica preprocesamiento dentro de cada fold: cada split calcula escalado/encoding solo con train, evitando contaminar test.

## 4. Si un modelo tiene mejor R² pero peor MAE, ¿cómo decides?
Depende de la métrica de negocio: si importa el error medio en unidades, prioriza MAE. Si buscas explicar varianza y los picos importan menos, puedes valorar R². Revisa también la distribución de errores por segmento.

## 5. En A/B: ¿qué significa que el IC95 de la diferencia incluya 0?
Que la diferencia verdadera podría ser cero (o negativa o positiva). No hay evidencia suficiente de efecto con 95% de confianza. No concluyas impacto.

## 6. Si p=0.03 pero el lift es 0.2%, ¿lanzas? ¿por qué?
No necesariamente. Es significativo, pero poco relevante. Compara 0.2 pp con el costo de implementar y el valor esperado. Si el costo supera el beneficio, no lances; mejor iterar o buscar un efecto mayor.

## 7. ¿Qué condiciones deben cumplirse para usar el z‑test de dos proporciones con tranquilidad?
Métrica binaria; aleatorización e independencia; tamaños suficientes (n*p y n*(1-p) ≳ 5–10 por grupo); misma ventana temporal/población comparable; sin peeking.

## 8. ¿Qué cambiarías si tu métrica fuera continua (ingreso por usuario) y no binaria?
Usaría t‑test (Welch si varianzas desiguales) o bootstrap para IC; reportaría diferencia media e IC95; validaría supuestos (CLT) o usaría métodos robustos/transformaciones si hay colas pesadas.