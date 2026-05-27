import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
# =============================
# CONFIGURACIÓN DE LA APP
# (debe ir primero en Streamlit)
# =============================
st.set_page_config(
    page_title="Predicción de Burnout",
    page_icon="🧠",
    layout="centered"
)
 
# =============================
# COLUMNAS ESPERADAS POR EL MODELO
# (mismo orden que en el entrenamiento)
# =============================
EXPECTED_COLS = [
    'age',
    'academic_year',
    'study_hours_per_day',
    'sleep_hours',
    'physical_activity',
    'screen_time',
    'internet_usage',
    'exam_pressure',
    'family_expectation',
    'financial_stress',
    'gender_Male',
    'gender_Other'
]
 
# =============================
# UMBRALES DE CLASIFICACIÓN
# IMPORTANTE: ajusta estos valores según la distribución
# real de predicciones de tu modelo. Puedes calcularlos así:
#   preds = model.predict(scaler.transform(X_test))
#   LOW  = np.percentile(preds, 33)
#   HIGH = np.percentile(preds, 66)
# Los valores por defecto (0.40 / 0.60) son más equilibrados
# que los originales (0.30 / 0.70) para modelos de regresión
# que comprimen las predicciones hacia el centro.
# =============================
LOW_THRESH  = 0.40
HIGH_THRESH = 0.60
 
# =============================
# CARGAR MODELO Y SCALER
# =============================
@st.cache_resource
def load_model():
    try:
        model  = joblib.load('modelo_burnout.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(
            f"No se encontró el archivo del modelo: {e}\n\n"
            "Asegúrate de que 'modelo_burnout.pkl' y 'scaler.pkl' "
            "estén en el mismo directorio que este script."
        )
        st.stop()
 
model, scaler = load_model()
 
# =============================
# TÍTULO Y DESCRIPCIÓN
# =============================
st.title("🧠 Predicción de Burnout Estudiantil")
st.write(
    "Ingresa la información del estudiante para predecir su nivel de burnout."
)
 
# =============================
# INPUTS DEL USUARIO
# =============================
col1, col2 = st.columns(2)
 
with col1:
    age = st.slider("Edad", 18, 35, 21)
    academic_year = st.slider("Año académico", 1, 5, 3)
    study_hours_per_day = st.slider("Horas de estudio por día", 0.0, 12.0, 5.0, step=0.5)
    sleep_hours = st.slider("Horas de sueño", 0.0, 12.0, 7.0, step=0.5)
    physical_activity = st.slider("Actividad física (0–10)", 0.0, 10.0, 3.0, step=0.5)
 
with col2:
    screen_time = st.slider("Tiempo en pantalla (horas)", 0.0, 14.0, 6.0, step=0.5)
    internet_usage = st.slider("Uso de internet (horas)", 0.0, 14.0, 5.0, step=0.5)
    exam_pressure = st.slider("Presión de exámenes (0–10)", 0.0, 10.0, 5.0, step=0.5)
    family_expectation = st.slider("Expectativa familiar (0–10)", 0.0, 10.0, 5.0, step=0.5)
    financial_stress = st.slider("Estrés financiero (0–10)", 0.0, 10.0, 5.0, step=0.5)
    gender = st.selectbox("Género", ["Female", "Male", "Other"])
 
# =============================
# ONE-HOT ENCODING MANUAL
# (Female es la categoría de referencia)
# =============================
gender_Male  = 1 if gender == "Male"  else 0
gender_Other = 1 if gender == "Other" else 0
 
# =============================
# DATAFRAME DE ENTRADA
# =============================
input_data = pd.DataFrame(
    {
        'age':                   [age],
        'academic_year':         [academic_year],
        'study_hours_per_day':   [study_hours_per_day],
        'sleep_hours':           [sleep_hours],
        'physical_activity':     [physical_activity],
        'screen_time':           [screen_time],
        'internet_usage':        [internet_usage],
        'exam_pressure':         [exam_pressure],
        'family_expectation':    [family_expectation],
        'financial_stress':      [financial_stress],
        'gender_Male':           [gender_Male],
        'gender_Other':          [gender_Other],
    }
)
 
# Reordenar columnas para que coincidan exactamente
# con las que vio el modelo durante el entrenamiento
input_data = input_data[EXPECTED_COLS]
 
# =============================
# PREDICCIÓN
# =============================
if st.button("Predecir Burnout", type="primary"):
    try:
        # Verificar que las columnas coinciden con el scaler
        if hasattr(scaler, 'feature_names_in_'):
            expected = list(scaler.feature_names_in_)
            actual   = list(input_data.columns)
            if expected != actual:
                st.error(
                    f"Las columnas no coinciden con las del scaler.\n\n"
                    f"Esperadas: {expected}\n\nRecibidas: {actual}"
                )
                st.stop()
 
        # Escalar y predecir
        scaled_data = scaler.transform(input_data)
        prediction  = model.predict(scaled_data)[0]
 
        # Limitar a rango [0, 1]
        prediction = float(np.clip(prediction, 0.0, 1.0))
 
        # ---- Resultado principal ----
        st.markdown("### Resultado")
 
        if prediction <= LOW_THRESH:
            st.success(f"🟢 Nivel de burnout **bajo**  —  score: `{prediction:.3f}`")
            st.progress(prediction)
        elif prediction <= HIGH_THRESH:
            st.warning(f"🟡 Nivel de burnout **moderado**  —  score: `{prediction:.3f}`")
            st.progress(prediction)
        else:
            st.error(f"🔴 Nivel de burnout **alto**  —  score: `{prediction:.3f}`")
            st.progress(prediction)
 
        # ---- Interpretación de factores de riesgo ----
        st.markdown("#### Factores de riesgo detectados")
 
        risk_factors = []
        if study_hours_per_day >= 9:
            risk_factors.append("⚠️ Horas de estudio muy altas (≥ 9 h/día)")
        if sleep_hours <= 5:
            risk_factors.append("⚠️ Pocas horas de sueño (≤ 5 h)")
        if exam_pressure >= 8:
            risk_factors.append("⚠️ Presión de exámenes muy elevada (≥ 8/10)")
        if financial_stress >= 8:
            risk_factors.append("⚠️ Estrés financiero muy elevado (≥ 8/10)")
        if family_expectation >= 8:
            risk_factors.append("⚠️ Expectativa familiar muy alta (≥ 8/10)")
        if physical_activity <= 2:
            risk_factors.append("⚠️ Actividad física muy baja (≤ 2/10)")
        if screen_time >= 10:
            risk_factors.append("⚠️ Tiempo de pantalla excesivo (≥ 10 h)")
 
        if risk_factors:
            for rf in risk_factors:
                st.write(rf)
        else:
            st.write("✅ No se detectaron factores de riesgo extremos.")
 
        # ---- Expander de diagnóstico técnico ----
        with st.expander("🔍 Detalle técnico (debug)"):
            st.write(f"**Score crudo del modelo:** `{prediction:.6f}`")
            st.write(f"**Umbrales aplicados:** bajo ≤ {LOW_THRESH} | alto > {HIGH_THRESH}")
            st.write("**Features enviadas al modelo:**")
            st.dataframe(input_data)
            st.write("**Valores escalados:**")
            st.dataframe(
                pd.DataFrame(scaled_data, columns=EXPECTED_COLS)
            )
 
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.info(
            "Revisa que el modelo y el scaler fueron entrenados "
            "con las mismas columnas que se están enviando ahora."
        )
 
# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "Aplicación desarrollada para el proyecto final de Analítica de Datos con IA.  \n"
    f"Umbrales de clasificación: bajo ≤ {LOW_THRESH} · moderado ≤ {HIGH_THRESH} · alto > {HIGH_THRESH}"
)
