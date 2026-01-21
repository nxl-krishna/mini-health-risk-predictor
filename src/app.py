import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Health Risk Predictor", layout="centered")

# --- ROBUST MODEL LOADING ---
@st.cache_resource
def load_artifacts():
    # 1. Get the directory where THIS file (app.py) lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construct the path to model.pkl (one folder up from src)
    model_path = os.path.join(current_dir, '..', 'model.pkl')
    
    # 3. Check if file exists before trying to load
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model at: {model_path}. \n\nMake sure you ran 'src/model_train.py' first!")
        
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['model'], artifacts['scaler']

# --- MAIN APP LOGIC ---
try:
    # Load model
    model, scaler = load_artifacts()

    st.title("🩺 Health Risk Predictor (Diabetes)")
    st.markdown("Enter the patient's details below to predict diabetes risk.")

    # Create Input Fields
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=33)

    # Prediction Logic
    # Prediction Logic
    if st.button("Predict Risk", type="primary"):
        # 1. Define the exact column names used during training
        # Note: These must match the CSV file headers exactly!
        col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # 2. Create a DataFrame instead of a NumPy array
        input_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, 
                                    insulin, bmi, dpf, age]], 
                                    columns=col_names)
        
        # 3. Scale input (Warning will disappear now)
        input_scaled = scaler.transform(input_data)
        
        # 4. Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        risk_score = probability[0][1] * 100
        
        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk Detected** (Confidence: {risk_score:.1f}%)")
            st.info("Recommendation: Please consult a healthcare professional.")
        else:
            st.success(f"✅ **Low Risk Detected** (Confidence: {100-risk_score:.1f}%)")

except Exception as e:
    # This block prevents the White Screen of Death
    st.error("🚨 An error occurred!")
    st.code(str(e))
    st.markdown("Check your terminal for more details.")