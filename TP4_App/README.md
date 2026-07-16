# 🧠 Predicción de Riesgo de ACV — Grupo 7 (TP4)

Aplicación web interactiva que despliega el modelo de **clasificación de riesgo de ACV
(stroke)** desarrollado en el proyecto integrador de *Inteligencia Artificial y Aprendizaje
Automático I* (UCA, 2026).

**Integrantes:** Formenti, Lucia · Montero, Juan Cruz · Do Brito, Franco · Ortiz, Victoria

---

## ¿Qué hace?

El usuario ingresa datos clínicos de un paciente (edad, glucosa, BMI, hipertensión,
hábitos, etc.) y la aplicación devuelve la **probabilidad estimada de sufrir un ACV**,
junto con:

- Un **medidor (gauge)** y una clasificación de riesgo (bajo / moderado / elevado).
- La **comparación con la prevalencia poblacional** (riesgo relativo).
- Un panel de **explicabilidad**: qué factores empujan el resultado hacia "ACV" o "no ACV".
- **Predicción en lote**: subir un CSV con varios pacientes y descargar los resultados.

## Modelo desplegado

- **Algoritmo:** Regresión Logística con `class_weight="balanced"` (elegido en el TP3).
- **Por qué:** es el modelo con **mayor recall (≈0.84)** y **AUC (≈0.84)** sobre la clase
  positiva, además de ser **interpretable** — ideal para una herramienta de *screening*
  donde lo crítico es no omitir pacientes en riesgo.
- **Métricas (test):** AUC 0.844 · Recall 0.84 · Precisión 0.134 · F1 0.231.
- El modelo prioriza el recall, por lo que **genera falsas alarmas a propósito** (precisión
  baja) para minimizar los falsos negativos.

## Estructura del repositorio

```
TP4_App/
├── app.py                              # Aplicación Streamlit
├── train_model.py                      # Entrena y serializa el modelo (+ feature engineering)
├── stroke_model.joblib                 # Modelo serializado (pipeline + metadatos)
├── healthcare-dataset-stroke-data.csv  # Dataset crudo (para reentrenar)
├── requirements.txt                    # Dependencias
└── README.md
```

## Ejecución local

```bash
pip install -r requirements.txt
# (opcional) reentrenar el modelo desde el CSV crudo:
python train_model.py
# levantar la app:
streamlit run app.py
```

Luego abrí `http://localhost:8501` en el navegador.

## Despliegue público

Ver **`GUIA_DESPLIEGUE.md`** para el paso a paso en *Streamlit Community Cloud* + GitHub.

---

> ⚠️ **Uso educativo.** Esta herramienta es un trabajo académico y **no constituye un
> diagnóstico médico**. El dataset no incluye variables clave (dieta, actividad física,
> antecedentes familiares), por lo que el resultado es orientativo.
