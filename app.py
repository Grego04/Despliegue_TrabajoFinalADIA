import streamlit as st
import pandas as pd
import joblib
import os

# Define the path to your models
folder_path = '/content/drive/MyDrive/Ejercicios análisis de datos IA/Despliegue EcoRide '
pipeline_path = os.path.join(folder_path, 'pipeline_preproc.pkl')
model_path = os.path.join(folder_path, 'modelo_churn.pkl')

# Load the pipeline and model
@st.cache_resource # Cache the loaded resources
def load_models():
    try:
        pipeline_preproc = joblib.load(pipeline_path)
        modelo_churn = joblib.load(model_path)
        return pipeline_preproc, modelo_churn
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the files exist and are correctly formatted.")
        st.stop()

pipeline_preproc, modelo_churn = load_models()

# --- Streamlit App Layout ---
st.set_page_config(page_title="EcoRide Churn Predictor", page_icon="🚴")

st.title("🚴 EcoRide Churn Prediction App")
st.write("Enter customer details to predict if they will churn.")

# Define input features based on typical churn prediction models
st.subheader("Customer Information")

edad = st.slider("Edad", 18, 70, 30)
gender = st.selectbox("Gender", ['Male', 'Female'])
dias_antiguedad = st.slider("Dias_Antiguedad", 0, 3650, 365) # Assuming days of service
suscripcion_plan = st.selectbox("Plan", ['Basic', 'Premium', 'Pro']) # Example plans, adjust as needed
gasto_promedio = st.number_input("Gasto_Promedio (€)", min_value=0.0, value=50.0, step=1.0)
soporte_tickets = st.slider("Soporte_Tickets", 0, 10, 1)
uso_mensual_km = st.number_input("Uso_Mensual_Km", min_value=0.0, value=100.0, step=1.0)
region = st.selectbox("Region", ['North', 'South', 'East', 'West']) # Example regions, adjust as needed


# Create a DataFrame for prediction with correct column names
input_data = pd.DataFrame({
    'Edad': [edad],
    'Gender': [gender],
    'Dias_Antiguedad': [dias_antiguedad],
    'Plan': [suscripcion_plan],
    'Gasto_Promedio': [gasto_promedio],
    'Soporte_Tickets': [soporte_tickets],
    'Uso_Mensual_Km': [uso_mensual_km],
    'Region': [region]
})

# Prediction button
if st.button("Predict Churn"):
    try:
        # Preprocess the input data using the loaded pipeline
        # NOTE: The column names in input_data must match the feature names
        # that your pipeline_preproc was trained on.
        processed_input = pipeline_preproc.transform(input_data)

        # Make prediction
        prediction = modelo_churn.predict(processed_input)
        prediction_proba = modelo_churn.predict_proba(processed_input)[:, 1] # Probability of churn

        if prediction[0] == 1: # Assuming 1 means churn
            st.error(f"Prediction: This customer is likely to CHURN (Probability: {prediction_proba[0]:.2f})")
        else:
            st.success(f"Prediction: This customer is unlikely to churn (Probability: {prediction_proba[0]:.2f})")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}. Please check your input data and model compatibility.")

st.markdown("""
---
*This app uses your custom preprocessor and churn prediction model.*
""")
