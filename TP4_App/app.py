"""
TP4 - Aplicación web de estimación de riesgo de ACV (stroke).
Grupo 7 - IA y Aprendizaje Automático I (UCA).

Versión rediseñada: interfaz clara, en español y con estética clínica/serena.
Objetivo: que cualquier persona (sin conocimientos técnicos) entienda el resultado,
informándose con calma y sin generar miedo. No reemplaza el diagnóstico médico.

Ejecutar localmente:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

from train_model import build_features  # misma ingeniería de features que el entrenamiento

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Estimador de riesgo de ACV — Grupo 7",
                   page_icon="🩺", layout="wide")

# ------------------------------------------------------------------ Paleta clínica
TEAL    = "#2C7A7B"   # color principal (verde azulado, "médico")
TEAL_D  = "#255E60"   # variante oscura para títulos
INK     = "#233240"   # texto
C_BAJO  = "#2F855A"   # verde sereno
C_MED   = "#B07D1A"   # ámbar sobrio
C_ALTO  = "#BC6A5A"   # coral apagado (atención SIN alarma; nada de rojo puro)
BG      = "#F3F8F9"   # fondo muy claro

# Plantilla común para todos los gráficos (fondo claro, texto oscuro y legible)
PLOT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, size=15, family="Helvetica, Arial, sans-serif"),
    margin=dict(t=50, b=20, l=10, r=10),
)

st.markdown(f"""
<style>
.stApp {{ background:{BG}; }}
/* Letras más grandes y legibles en toda la app */
html, body, [class*="css"], .stMarkdown, p, li, label {{
    font-size: 17px; color:{INK};
}}
h1,h2,h3,h4, [data-testid="stHeading"] * {{ color:{TEAL_D} !important; font-weight:700 !important; }}
/* Encabezado tipo tarjeta */
.banner {{
    background:#ffffff; border-left:9px solid {TEAL};
    padding:22px 28px; border-radius:14px;
    box-shadow:0 3px 16px rgba(37,94,96,0.10); margin-bottom:14px;
}}
.banner h1 {{ margin:0; font-size:32px; color:{TEAL_D} !important; }}
.banner p  {{ margin:10px 0 0 0; font-size:17px; color:{INK}; line-height:1.5; }}
/* Botón principal */
div.stButton > button {{
    background:{TEAL}; color:#ffffff !important; border:0; border-radius:11px;
    font-weight:700; font-size:19px; padding:13px 0;
}}
div.stButton > button:hover {{ background:{TEAL_D}; color:#ffffff !important; }}
/* Etiquetas de los campos en negrita */
label, [data-testid="stWidgetLabel"] p {{ font-weight:600 !important; color:{INK} !important; font-size:16px !important; }}
/* Cuadro de resultado */
.resultado {{
    background:#ffffff; border-radius:14px; padding:20px 24px;
    box-shadow:0 3px 16px rgba(37,94,96,0.10); border:1px solid #E2ECEE; margin-bottom:8px;
}}
.resultado h2 {{ margin:0 0 6px 0; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join(BASE_DIR, "stroke_model.joblib"))

bundle = load_bundle()
model = bundle["model"]
THRESH = 0.5                      # umbral del modelo (ya reponderado)
base = bundle["positive_rate"]    # prevalencia poblacional de referencia

# --------------------------------------------------- Opciones en español -> valor del modelo
GENERO  = {"Femenino": "Female", "Masculino": "Male"}
SI_NO   = {"Sí": "Yes", "No": "No"}
RESID   = {"Urbana": "Urban", "Rural": "Rural"}
TRABAJO = {"Empleo privado": "Private", "Trabajo independiente": "Self-employed",
           "Empleo público": "Govt_job", "Es menor / no trabaja": "children",
           "Nunca trabajó": "Never_worked"}
FUMADOR = {"Nunca fumó": "never smoked", "Ex fumador/a": "formerly smoked",
           "Fuma actualmente": "smokes", "Prefiero no decir / no sé": "Unknown"}

ETIQUETAS = {
    "age": "Edad", "avg_glucose_level": "Nivel de glucosa", "bmi": "Índice de masa corporal (IMC)",
    "hypertension": "Presión alta", "heart_disease": "Enfermedad del corazón",
    "gender": "Género (masculino)", "ever_married": "Estuvo casado/a",
    "Residence_type": "Vive en zona urbana",
    "work_type_Private": "Trabajo: privado", "work_type_Self-employed": "Trabajo: independiente",
    "work_type_children": "Es menor de edad", "work_type_Never_worked": "Nunca trabajó",
    "smoking_status_formerly smoked": "Ex fumador/a",
    "smoking_status_never smoked": "Nunca fumó", "smoking_status_smokes": "Fuma",
    "age_group_joven_adulto": "Grupo de edad: joven", "age_group_adulto": "Grupo de edad: adulto",
    "age_group_mayor": "Grupo de edad: mayor", "bmi_cat_normal": "IMC normal",
    "bmi_cat_sobrepeso": "IMC con sobrepeso", "bmi_cat_obeso": "IMC con obesidad",
}

# ------------------------------------------------------------------ Encabezado
st.markdown(f"""
<div class="banner">
  <h1>🩺 Estimador de riesgo de ACV</h1>
  <p>Una herramienta <b>orientativa</b> que, a partir de algunos datos de salud, estima
  si la probabilidad de sufrir un <b>accidente cerebrovascular (ACV)</b> es más alta o más
  baja que el promedio de la población. <b>Es información, no un diagnóstico.</b></p>
</div>
""", unsafe_allow_html=True)

# Mensaje tranquilizador (sin generar miedo)
st.info(
    "ℹ️ **Leé esto con calma.** Este resultado es puramente **estadístico**: sale de un "
    "modelo entrenado con datos de miles de personas. Cada ser humano es diferente, así que "
    "un valor **alto no significa** que vayas a tener un ACV, y uno **bajo no lo descarta**. "
    "La idea es **informarte, no asustarte**. Si algo te preocupa, lo mejor siempre es "
    "conversarlo con un profesional de la salud de tu confianza. 🌿"
)

# Sección opcional: cómo funciona y qué tan confiable es (en lenguaje simple)
with st.expander("🔎 ¿Cómo funciona y qué tan confiable es esta estimación?"):
    m = bundle["metrics"]
    st.markdown(f"""
Esta herramienta aprendió a partir de **más de 5.000 registros reales de pacientes**.
Para cada persona mira datos como la edad, la presión, la glucosa o el IMC y estima su
**nivel de riesgo** comparado con el resto.

- **Detecta a alrededor de {int(round(m['Recall_ACV']*100))} de cada 100 personas** que sí
  tuvieron un ACV. Preferimos que *"suene la alarma de más"* antes que dejar pasar un caso.
- Como contrapartida, **también marca riesgo en varias personas que nunca lo tendrán**. Por eso,
  un resultado alto hay que tomarlo con calma y, si querés, consultarlo con un médico.
- En general **distingue bastante bien** a quienes tuvieron un ACV de quienes no
  (su puntaje de acierto global, llamado *AUC*, es {str(m['AUC']).replace('.',',')} sobre 1;
  0,5 sería puro azar).

En resumen: es una buena **orientación**, no una certeza.
""")

st.divider()

# ------------------------------------------------------------------ Datos del paciente
st.subheader("📋 Completá los datos")
st.caption("Movés los controles o elegís la opción que corresponda. Al final, tocá **Calcular**.")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Edad (años)", 0, 100, 55)
    gender_es = st.selectbox("Género", list(GENERO.keys()))
    married_es = st.selectbox("¿Estuvo casado/a alguna vez?", list(SI_NO.keys()))
with col2:
    hypertension_es = st.selectbox("¿Tiene presión alta (hipertensión)?", ["No", "Sí"])
    heart_es = st.selectbox("¿Tiene alguna enfermedad del corazón?", ["No", "Sí"])
    residence_es = st.selectbox("¿Dónde vive?", list(RESID.keys()))
with col3:
    avg_glucose = st.slider("Nivel de glucosa en sangre (mg/dL)", 50.0, 280.0, 105.0, step=0.5,
                            help="Es el azúcar en sangre. En ayunas, un valor normal ronda 70–100 mg/dL.")
    bmi = st.slider("Índice de masa corporal – IMC", 10.0, 60.0, 28.0, step=0.1,
                    help="Relaciona peso y altura. Entre 18,5 y 25 se considera peso normal; más de 30, obesidad.")
    work_es = st.selectbox("Situación laboral", list(TRABAJO.keys()))
smoking_es = st.selectbox("¿Fuma o fumó?", list(FUMADOR.keys()))

# Traducción a los valores que espera el modelo
gender = GENERO[gender_es]; ever_married = SI_NO[married_es]
residence = RESID[residence_es]; work_type = TRABAJO[work_es]; smoking = FUMADOR[smoking_es]

# Validación de entradas
errores = []
if age <= 0: errores.append("La edad debe ser mayor a 0.")
if not (50 <= avg_glucose <= 280): errores.append("La glucosa debe estar entre 50 y 280 mg/dL.")
if not (10 <= bmi <= 60): errores.append("El IMC debe estar entre 10 y 60.")
if age < 18 and work_type not in ["children", "Never_worked"]:
    st.warning("Para menores de 18 lo habitual es 'Es menor / no trabaja' o 'Nunca trabajó'. "
               "Revisá la situación laboral.")

st.text("")  # respiro visual

# ------------------------------------------------------------------ Cálculo y resultado
if st.button("🔎 Calcular mi nivel de riesgo", type="primary", use_container_width=True):
    if errores:
        for e in errores:
            st.error(e)
    else:
        raw = dict(gender=gender, age=age,
                   hypertension=1 if hypertension_es == "Sí" else 0,
                   heart_disease=1 if heart_es == "Sí" else 0,
                   ever_married=ever_married, work_type=work_type,
                   Residence_type=residence, avg_glucose_level=avg_glucose,
                   bmi=bmi, smoking_status=smoking)
        X = build_features(raw, bundle["bmi_median"], bundle["glu_cap"], bundle["bmi_cap"])
        proba = float(model.predict_proba(X)[0, 1])
        puntaje = proba * 100

        # Nivel + color sereno + mensaje que informa y acompaña (sin asustar)
        if proba >= THRESH:
            nivel, color = "más alto que el promedio", C_ALTO
            mensaje = ("Tu perfil se ubica en un nivel de riesgo **más alto que el promedio**. "
                       "Esto **no** quiere decir que vayas a tener un ACV: es una señal para "
                       "prestar atención a los factores que sí se pueden cuidar —presión, glucosa, "
                       "peso y tabaquismo— y, si te quedás más tranquilo/a, conversarlo con un médico.")
        elif proba >= base * 2:
            nivel, color = "intermedio", C_MED
            mensaje = ("Tu perfil muestra un nivel de riesgo **intermedio**. La gran mayoría de las "
                       "personas en esta franja **no** tienen un ACV; aun así, cuidar los hábitos "
                       "de salud siempre suma.")
        else:
            nivel, color = "bajo", C_BAJO
            mensaje = ("Tu perfil se ubica en un nivel de riesgo **bajo** respecto del promedio. "
                       "Es una buena noticia; mantener hábitos saludables es la mejor forma de "
                       "que siga así.")

        st.divider()
        st.markdown(
            f"<div class='resultado'><h2 style='color:{color}!important;'>Nivel de riesgo estimado: "
            f"{nivel}</h2><p style='font-size:17px;margin:0;'>{mensaje}</p></div>",
            unsafe_allow_html=True)

        col_g, col_f = st.columns([1, 1])

        # --- Medidor (gauge) ---
        with col_g:
            st.markdown("#### Tu puntaje de riesgo")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=puntaje,
                number={"suffix": " / 100", "font": {"size": 40, "color": INK}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": INK},
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, base * 200], "color": "#DCEFE4"},   # zona baja (verde suave)
                        {"range": [base * 200, 50], "color": "#F7ECD2"},  # zona media (ámbar suave)
                        {"range": [50, 100], "color": "#F3DDD7"},         # zona alta (coral suave)
                    ],
                }))
            fig.update_layout(height=270, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Es un **indicador de 0 a 100**: cuanto más alto, más factores de riesgo detectó el "
                "modelo. Las franjas de color (verde → ámbar → coral) marcan zonas de riesgo bajo, "
                "intermedio y alto. No es la probabilidad médica exacta de tener un ACV, sino una "
                "referencia comparativa.")

        # --- Factores que más influyeron ---
        with col_f:
            st.markdown("#### ¿Qué pesó en tu resultado?")
            scaler = model.named_steps["scaler"]; clf = model.named_steps["clf"]
            x_std = (X.values[0] - scaler.mean_) / scaler.scale_
            contrib = clf.coef_[0] * x_std
            cs = pd.Series(contrib, index=bundle["feature_columns"]).sort_values()
            top = pd.concat([cs.head(5), cs.tail(5)])
            nombres = [ETIQUETAS.get(i, i) for i in top.index]
            colores = [C_BAJO if v < 0 else C_ALTO for v in top.values]
            figb = go.Figure(go.Bar(x=top.values, y=nombres, orientation="h",
                                    marker_color=colores))
            figb.update_layout(height=300, xaxis_title="", **PLOT_LAYOUT)
            figb.update_xaxes(showticklabels=False, zeroline=True, zerolinecolor="#B9C6C9")
            st.plotly_chart(figb, use_container_width=True)
            st.markdown(
                f"<p style='font-size:14px;color:{INK};'>Estos son los datos que "
                f"<b>más influyeron en tu resultado</b>. En <b style='color:{C_BAJO}'>verde</b>, "
                f"los que <b>bajan</b> el riesgo; en <b style='color:{C_ALTO}'>coral</b>, los que lo "
                f"<b>suben</b>. Por ejemplo, más edad o glucosa alta suelen empujar hacia arriba.</p>",
                unsafe_allow_html=True)

        st.success(
            "🌿 Recordá: esto es una orientación estadística, no un diagnóstico. Muchos de los "
            "factores que más influyen se pueden mejorar con hábitos y control médico. "
            "Ante cualquier duda, consultá con un profesional de la salud.")

# ------------------------------------------------------------------ Opción avanzada: varios pacientes
st.divider()
with st.expander("📂 Opción avanzada: analizar muchas personas a la vez (archivo CSV)"):
    st.caption("Pensado para uso técnico. Subí un archivo CSV con una fila por persona y estas "
               "columnas (con los valores del dataset original de Kaggle, en inglés): gender, age, "
               "hypertension, heart_disease, ever_married, work_type, Residence_type, "
               "avg_glucose_level, bmi, smoking_status.")
    up = st.file_uploader("Archivo CSV", type="csv")
    if up is not None:
        dfin = pd.read_csv(up)
        try:
            Xs = pd.concat([build_features(r, bundle["bmi_median"], bundle["glu_cap"],
                                           bundle["bmi_cap"]) for r in dfin.to_dict("records")],
                           ignore_index=True)
            dfin["puntaje_riesgo"] = (model.predict_proba(Xs)[:, 1] * 100).round(1)
            dfin["nivel"] = np.where(dfin["puntaje_riesgo"] >= 50, "Más alto que el promedio",
                                     np.where(dfin["puntaje_riesgo"] >= base * 200, "Intermedio", "Bajo"))
            st.dataframe(dfin)
            st.download_button("⬇️ Descargar resultados",
                               dfin.to_csv(index=False).encode("utf-8"),
                               "resultados_riesgo_acv.csv", "text/csv")
        except Exception as ex:
            st.error(f"No se pudo procesar el archivo. Revisá que tenga las columnas indicadas. ({ex})")

st.divider()
st.caption("Grupo 7 — Formenti, Montero, Do Brito, Ortiz · UCA 2026 · Proyecto integrador de "
           "Inteligencia Artificial y Aprendizaje Automático I · Herramienta educativa, no diagnóstica.")
