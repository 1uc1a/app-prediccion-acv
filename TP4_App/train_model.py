"""
TP4 - Entrenamiento y serialización del modelo a desplegar.

Reproduce la ingeniería de características del TP1 a partir del CSV crudo de Kaggle
y entrena el modelo recomendado en el TP3: Regresión Logística balanceada (alto recall
y AUC, interpretable) para estimar el riesgo de ACV (stroke).

Salida: stroke_model.joblib  (pipeline + metadatos para la app)

Uso:  python train_model.py
Requiere 'healthcare-dataset-stroke-data.csv' en la misma carpeta (o ajustar RAW_PATH).
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

RAW_PATH = "healthcare-dataset-stroke-data.csv"
RANDOM_STATE = 42

# --- Categorías de referencia (drop_first) y bins, idénticos al TP1 ---
WORK_TYPES = ["Never_worked", "Private", "Self-employed", "children"]      # base: Govt_job
SMOKING = ["formerly smoked", "never smoked", "smokes"]                     # base: Unknown
AGE_BINS = [0, 18, 40, 60, 120]
AGE_LABELS = ["niño", "joven_adulto", "adulto", "mayor"]
BMI_BINS = [0, 18.5, 25, 30, 100]
BMI_LABELS = ["bajo_peso", "normal", "sobrepeso", "obeso"]

FEATURE_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "Residence_type", "avg_glucose_level", "bmi",
    "work_type_Never_worked", "work_type_Private", "work_type_Self-employed",
    "work_type_children", "smoking_status_formerly smoked",
    "smoking_status_never smoked", "smoking_status_smokes",
    "age_group_joven_adulto", "age_group_adulto", "age_group_mayor",
    "bmi_cat_normal", "bmi_cat_sobrepeso", "bmi_cat_obeso",
]

def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)

def build_features(raw_row: dict, bmi_median, glu_cap, bmi_cap):
    """Convierte una fila de inputs crudos en el vector de 21 features del TP1."""
    g_lo, g_hi = glu_cap; b_lo, b_hi = bmi_cap
    age = float(raw_row["age"])
    glu = float(np.clip(raw_row["avg_glucose_level"], g_lo, g_hi))
    bmi = raw_row.get("bmi", None)
    bmi = bmi_median if bmi is None or (isinstance(bmi, float) and np.isnan(bmi)) else bmi
    bmi = float(np.clip(bmi, b_lo, b_hi))

    f = {c: 0 for c in FEATURE_COLUMNS}
    f["gender"] = 1 if raw_row["gender"] == "Male" else 0
    f["age"] = age
    f["hypertension"] = int(raw_row["hypertension"])
    f["heart_disease"] = int(raw_row["heart_disease"])
    f["ever_married"] = 1 if raw_row["ever_married"] == "Yes" else 0
    f["Residence_type"] = 1 if raw_row["Residence_type"] == "Urban" else 0
    f["avg_glucose_level"] = glu
    f["bmi"] = bmi
    wt = raw_row["work_type"]
    if f"work_type_{wt}" in f: f[f"work_type_{wt}"] = 1
    sm = raw_row["smoking_status"]
    if f"smoking_status_{sm}" in f: f[f"smoking_status_{sm}"] = 1
    ag = pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS)[0]
    if f"age_group_{ag}" in f: f[f"age_group_{ag}"] = 1
    bc = pd.cut([bmi], bins=BMI_BINS, labels=BMI_LABELS)[0]
    if f"bmi_cat_{bc}" in f: f[f"bmi_cat_{bc}"] = 1
    return pd.DataFrame([f])[FEATURE_COLUMNS]

def main():
    raw = pd.read_csv(RAW_PATH).drop(columns=["id"])
    raw = raw[raw.gender != "Other"].copy()
    bmi_median = float(raw["bmi"].median())
    raw["bmi"] = raw["bmi"].fillna(bmi_median)
    glu_cap = iqr_bounds(raw["avg_glucose_level"])
    bmi_cap = iqr_bounds(raw["bmi"])

    # Construir matriz de features fila por fila (reutiliza build_features -> consistencia total)
    rows = [build_features(r, bmi_median, glu_cap, bmi_cap) for r in raw.to_dict("records")]
    X = pd.concat(rows, ignore_index=True)
    y = raw["stroke"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]; pred = model.predict(Xte)
    metrics = {
        "AUC": round(float(roc_auc_score(yte, proba)), 3),
        "Recall_ACV": round(float(recall_score(yte, pred)), 3),
        "Precision_ACV": round(float(precision_score(yte, pred, zero_division=0)), 3),
        "F1_ACV": round(float(f1_score(yte, pred)), 3),
    }
    # Reentrenar con TODO el dataset para el modelo final desplegado
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "bmi_median": bmi_median,
        "glu_cap": glu_cap,
        "bmi_cap": bmi_cap,
        "work_types": ["Govt_job"] + WORK_TYPES,
        "smoking": ["Unknown"] + SMOKING,
        "metrics": metrics,
        "positive_rate": round(float(y.mean()), 4),
    }
    joblib.dump(bundle, "stroke_model.joblib")
    print("Modelo serializado en stroke_model.joblib")
    print("Métricas en test:", json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()
