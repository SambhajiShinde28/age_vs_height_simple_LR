import streamlit as st
import numpy as np
import joblib

model = joblib.load("../model-files/LR_model1.joblib")

st.title("Height Predictor Based on Age")
st.write("Enter your age, and I’ll predict your height!")

age = st.number_input("Enter age:", min_value=1, max_value=100, value=20)

if st.button("Predict Height"):
    age_input = np.array([[age]])
    predicted_height = model.predict(age_input.reshape(-1, 1))
    
    st.success(f"Predicted Height: {predicted_height[0]:.2f} cm")
