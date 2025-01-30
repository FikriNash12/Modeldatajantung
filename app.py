import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  

# Load the trained model
with open("model.pkl", "rb") as file:
    model = joblib.load(file)  

st.title("Heart Failure Prediction Model")

# Define categorical and numerical features with new naming
categorical_cols = ['cat__Gender', 'cat__Region', 'cat__Smoking_History', 'cat__Diabetes_History', 
                    'cat__Hypertension_History', 'cat__Physical_Activity', 'cat__Diet_Quality', 
                    'cat__Alcohol_Consumption', 'cat__Family_History']

numerical_cols = ['num__Age', 'num__Cholesterol_Level', 'num__BMI', 'num__Heart_Rate', 
                  'num__Systolic_BP', 'num__Diastolic_BP', 'num__Stress_Levels']

# User inputs for categorical features
gender = st.selectbox("Gender", ['Female', 'Male'])
region = st.selectbox("Region", ['Rural', 'Urban'])
smoking_history = st.selectbox("Smoking History", ["No", "Yes"])
diabetes_history = st.radio("Diabetes History", ["No", "Yes"])
hypertension_history = st.radio("Hypertension History", ["No", "Yes"])
physical_activity = st.selectbox("Physical Activity Level", ['High', 'Low', 'Moderate'])
diet_quality = st.selectbox("Diet Quality", ['Average', 'Good', 'Poor'])
alcohol_consumption = st.selectbox("Alcohol Consumption", ['High', 'Low', 'Moderate'])
family_history = st.radio("Family History of Disease", ["No", "Yes"])

# User inputs for numerical features
age = st.number_input("Age", min_value=0, max_value=120, step=1)
cholesterol_level = st.number_input("Cholesterol Level", min_value=100, max_value=350, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, step=1)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=120, step=1)
stress_levels = st.slider("Stress Levels (1-10)", min_value=1, max_value=10, step=1)

# Create DataFrame
input_data = pd.DataFrame([[gender, region, smoking_history, diabetes_history, hypertension_history, 
                            physical_activity, diet_quality, alcohol_consumption, family_history,
                            age, cholesterol_level, bmi, heart_rate, systolic_bp, diastolic_bp, stress_levels]], 
                          columns=categorical_cols + numerical_cols)

# Encoding categorical features
encoding_map = {
    'cat__Gender': {'Female': 0, 'Male': 1},
    'cat__Region': {'Rural': 0, 'Urban': 1},
    'cat__Smoking_History': {'No': 0, 'Yes': 1},
    'cat__Diabetes_History': {'No': 0, 'Yes': 1},
    'cat__Hypertension_History': {'No': 0, 'Yes': 1},
    'cat__Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2},
    'cat__Diet_Quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'cat__Alcohol_Consumption': {'Low': 0, 'Moderate': 1, 'High': 2},
    'cat__Family_History': {'No': 0, 'Yes': 1}
}

# Apply encoding
for col in categorical_cols:
    if col in input_data.columns:
        input_data[col] = input_data[col].map(encoding_map[col])
    else:
        st.write(f"⚠️ Warning: Column '{col}' is missing!")

# Convert numerical inputs to float
for col in numerical_cols:
    try:
        input_data[col] = input_data[col].astype(float)
    except ValueError:
        st.write(f"⚠️ Warning: Could not convert {col} to float. Setting to 0.")
        input_data[col] = 0.0

# Scaling numerical features
scaler = StandardScaler()
input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

# Ensure column order matches model expectations
expected_features = model.feature_names_in_
input_data = input_data[expected_features]

# Make prediction
if st.button("Predict"):
    try:
        # Check input shape
        st.write(f"Input data shape: {input_data.shape}")
        
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
        else:
            st.write("Model does not have a 'predict' method.")
            
    except Exception as e:
        st.write(f"Error: {e}")
