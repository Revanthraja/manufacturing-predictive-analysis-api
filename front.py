import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.title("Predictive Analysis for Manufacturing Operations")

# Upload Dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload your manufacturing dataset (CSV)", type="csv")
if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/upload", files=files)
    if response.status_code == 200:
        st.success("Dataset uploaded successfully!")
    else:
        st.error(response.json()["detail"])

# Train Model
st.header("2. Train the Model")
if st.button("Train Model"):
    response = requests.post(f"{API_URL}/train")
    if response.status_code == 200:
        metrics = response.json()
        st.success(f"Model trained successfully! Accuracy: {metrics['accuracy']:.2f}, F1 Score: {metrics['f1_score']:.2f}")
    else:
        st.error(response.json()["detail"])

# Make Predictions
st.header("3. Make Predictions")
with st.form("prediction_form"):
    temperature = st.number_input("Temperature", min_value=50.0, max_value=100.0, step=0.1)
    run_time = st.number_input("Run Time", min_value=50.0, max_value=200.0, step=0.1)
    shift = st.selectbox("Shift", ["Morning", "Evening", "Night"])
    machine_type = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
    submit = st.form_submit_button("Predict Downtime")

if submit:
    payload = {
        "Temperature": temperature,
        "Run_Time": run_time,
        "Shift": shift,
        "Machine_Type": machine_type
    }
    response = requests.post(f"{API_URL}/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: **Downtime: {result['Downtime']}**, Confidence: **{result['Confidence']:.2f}**")
    else:
        st.error(response.json()["detail"])
