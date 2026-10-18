import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction System")
st.write("Enter the four patient attributes below to get a model prediction.")
st.warning("Educational demo only. This prediction is not a medical diagnosis.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4])
thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=50, max_value=250, value=150)

if st.button("Predict Heart Disease", type="primary"):
    input_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Model prediction: Heart disease indicated")
    else:
        st.success("Model prediction: No heart disease indicated")
