
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define the base path for loading the model and scaler
BASE_PATH_APP = '/content/drive/MyDrive/Trabajo Final IA/Despliegue_burnout/'

# Load the model and scaler
try:
    scaler = joblib.load(os.path.join(BASE_PATH_APP, 'scaler.pkl'))
    model = joblib.load(os.path.join(BASE_PATH_APP, 'modelo_burnout.pkl'))
except Exception as e:
    st.error(f"Error loading model or scaler. Make sure the files are in the correct path: {e}")
    st.stop()

st.title("Burnout Score Prediction App")
st.write("Enter the student's characteristics to predict the burnout score:")

# Input features based on the provided list
# Ensure the order of input widgets corresponds to the expected order of features
# in the model after one-hot encoding.

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=20, help="Age of the student")
    academic_year = st.number_input("Academic Year", min_value=1, max_value=7, value=3, help="Academic year (e.g., 1st, 2nd, 3rd)")
    study_hours_per_day = st.number_input("Study Hours Per Day", min_value=0.0, max_value=16.0, value=4.0, step=0.5, help="Average hours spent studying per day")
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5, help="Average hours of sleep per night")
    physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=20.0, value=3.0, step=0.5, help="Hours of physical activity per week")
    
with col2:
    screen_time = st.number_input("Screen Time (hours/day)", min_value=0.0, max_value=18.0, value=6.0, step=0.5, help="Hours spent on screens per day (non-academic)")
    internet_usage = st.number_input("Internet Usage (hours/day)", min_value=0.0, max_value=18.0, value=5.0, step=0.5, help="Hours of internet usage per day (non-academic)")
    exam_pressure = st.slider("Exam Pressure (1-10)", min_value=1, max_value=10, value=5, help="Perceived pressure from exams")
    family_expectation = st.slider("Family Expectation (1-10)", min_value=1, max_value=10, value=5, help="Perceived family expectations")
    financial_stress = st.slider("Financial Stress (1-10)", min_value=1, max_value=10, value=5, help="Perceived financial stress")
    gender_selection = st.selectbox("Gender", ['Male', 'Female', 'Other'], help="Select student's gender")


# Prepare gender for one-hot encoding
gender_Male = 1 if gender_selection == 'Male' else 0
gender_Other = 1 if gender_selection == 'Other' else 0

# Create a DataFrame from inputs with EXACTLY the training column names and order
input_data = pd.DataFrame([[
    age,
    academic_year,
    study_hours_per_day,
    sleep_hours,
    physical_activity,
    screen_time,
    internet_usage,
    exam_pressure,
    family_expectation,
    financial_stress,
    gender_Male,
    gender_Other
]], 
columns=[
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
])

if st.button("Predict Burnout Score"):
    # Scale the input data using the loaded scaler
    scaled_input_data = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(scaled_input_data)[0]

    # Limit the result between 0 and 10
    final_prediction = np.clip(prediction, 0, 10)

    st.success(f"Predicted Burnout Score: {final_prediction:.2f}")
    st.balloons()

st.write("---")
st.markdown("**Disclaimer:** This is a demo application. The accuracy of predictions depends on the training data and features used. Please ensure the input features and their order match your trained model's requirements.")
