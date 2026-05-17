import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# CARGAR MODELO Y SCALER
# =============================

@st.cache_resource
def load_model():
    model = joblib.load('modelo_burnout.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# =============================
# CONFIGURACIÓN DE LA APP
# =============================

st.set_page_config(
    page_title="Predicción de Burnout",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Predicción de Burnout Estudiantil")
st.write(
    "Ingrese la información del estudiante para predecir el nivel de burnout."
)

# =============================
# INPUTS DEL USUARIO
# =============================

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad", 18, 35, 21)

    academic_year = st.slider(
        "Año Académico",
        1,
        5,
        3
    )

    study_hours_per_day = st.slider(
        "Horas de estudio por día",
        0.0,
        12.0,
        5.0
    )

    sleep_hours = st.slider(
        "Horas de sueño",
        0.0,
        12.0,
        7.0
    )

    physical_activity = st.slider(
        "Actividad física",
        0.0,
        10.0,
        3.0
    )

with col2:

    screen_time = st.slider(
        "Tiempo en pantalla",
        0.0,
        14.0,
        6.0
    )

    internet_usage = st.slider(
        "Uso de internet",
        0.0,
        14.0,
        5.0
    )

    exam_pressure = st.slider(
        "Presión de exámenes",
        0.0,
        10.0,
        5.0
    )

    family_expectation = st.slider(
        "Expectativa familiar",
        0.0,
        10.0,
        5.0
    )

    financial_stress = st.slider(
        "Estrés financiero",
        0.0,
        10.0,
        5.0
    )

    gender = st.selectbox(
        "Género",
        ["Female", "Male", "Other"]
    )

# =============================
# ONE HOT ENCODING MANUAL
# =============================

gender_Male = 1 if gender == "Male" else 0
gender_Other = 1 if gender == "Other" else 0

# =============================
# DATAFRAME DE ENTRADA
# =============================

input_data = pd.DataFrame({
    'age': [age],
    'academic_year': [academic_year],
    'study_hours_per_day': [study_hours_per_day],
    'sleep_hours': [sleep_hours],
    'physical_activity': [physical_activity],
    'screen_time': [screen_time],
    'internet_usage': [internet_usage],
    'exam_pressure': [exam_pressure],
    'family_expectation': [family_expectation],
    'financial_stress': [financial_stress],
    'gender_Male': [gender_Male],
    'gender_Other': [gender_Other]
})

# =============================
# PREDICCIÓN
# =============================
if st.button("Predecir Burnout"):

    try:

        # Escalar datos
        scaled_data = scaler.transform(input_data)

        # Predicción
        prediction = model.predict(scaled_data)[0]

        # Limitar valores entre 0 y 1
        prediction = np.clip(prediction, 0, 1)

        # Mostrar resultado
        st.success(
            f"Nivel de burnout predicho: {prediction:.2f}"
        )

        # Clasificación
        if prediction <= 0.3:
            st.info("🟢 Nivel de burnout bajo")

        elif prediction <= 0.7:
            st.warning("🟡 Nivel de burnout moderado")

        else:
            st.error("🔴 Nivel de burnout alto")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
# =============================
# FOOTER
# =============================

st.markdown("---")

st.write(
    "Aplicación desarrollada para el proyecto final de Analítica de Datos con IA."
)
