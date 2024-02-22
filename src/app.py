from src.pipeline.utils import load_pkl
import streamlit as st
import numpy as np

# File path / load models
model_file_path = 'models/model.pkl'

# Model
model = load_pkl(model_file_path)

# User input gets stored here
inputs = []

# Streamlit application
st.title('Heart attack prediction')

## Form
with st.form("my_form"):
    
    age = st.slider("Select age:", 0, 120, 20)
    
    gender = st.selectbox(
        "What gender?",
        ("Male", "Female"),
        index=None,
        placeholder="Please select a gender..."
    )
    
    # Convert to num
    if gender == "Female":
        gender = 0
    else:
        gender = 1
    
    rbp = st.number_input("Insert resting blood pressure:")
    
    serum_chl = st.number_input("Serum Cholesterol in mg/dL (e.g., 212):")
    
    fbs = st.number_input("Insert fasting blood sugar in mg/dL (e.g, 120):")
    
    # Convert to number
    if fbs <= 120:
        fbs = 0
    else:
        fbs = 1
    
    resting_ecg = st.selectbox(
        "Select resting electrocardiographic results:",
        ("Normal", "Abnormality", "Hypertrophy"),
        index=None,
        placeholder="Please select one..."
    )
    
    # Convert
    if resting_ecg == "Normal":
        resting_ecg = 0
    elif resting_ecg == "Abnormality":
        resting_ecg = 1
    else:
        resting_ecg = 2
    
    max_heart_rate = st.number_input("Max heart rate achieved:")
    
    exercise_angina = st.selectbox(
        "Exercised induced angina?",
        ("Yes", "No"),
        index=None,
        placeholder="Please select one..."
    )
    
    # Yes / No
    if exercise_angina == "No":
        exercise_angina = 0
    else:
        exercise_angina = 1
    
    oldpeak = st.number_input("Oldpeak (ST Depression):")
    
    slope_peak = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ("Upsloping", "Flat", "Downsloping"),
        index=None,
        placeholder="Please choose one..."
    )
    
    # Slope peak
    if slope_peak == "Upsloping":
        slope_peak = 0
    elif slope_peak == "Flat":
        slope_peak = 1
    else:
        slope_peak = 2
    
    num_mv = st.number_input(
        "Number of Major Vessels (0 to 3):",
        min_value=0,
        max_value=3
    )
    
    thal = st.selectbox(
        "Thalassemia",
        ("Normal", "Fixed Defect", "Reversible Defect"),
        index=None,
        placeholder="Please choose one..."
    )
    
    target = st.number_input(
        "Target",
        min_value=0.1,
        max_value=0.9
    )
    
    # Thalassemia
    if thal == "Normal":
        thal = 0
    elif thal == "Fixed Defect":
        thal = 1
    else:
        thal = 2
    
    submitted = st.form_submit_button("Submit")
    form_list = [age, gender, rbp, serum_chl, fbs, resting_ecg,
                 max_heart_rate, exercise_angina, oldpeak, slope_peak,
                 num_mv, thal, target]
    
    if submitted:
        # Append to list
        input_list = np.array(form_list).reshape(1,-1)
        
        # Predict model
        result = model.predict(input_list)
        
        # Display 
        if result == 0:
            st.header("Typical Angina")
        elif result == 1:
            st.header("Atypical Angina")
        elif result == 2: 
            st.header("Non-anginal Pain")
        else:
            st.header("Asymptomatic")