import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Setup ---
st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details below or click the sample button to auto-fill test values.")

# --- Model Check ---
model_path = "models/xgboost_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please run `fraud_detection.py` to train and save the model first.")
    st.stop()  # ⛔ Prevent app from continuing
else:
    model = joblib.load(model_path)

# --- Sample Values ---
inputs = [0.0] * 30  # Default values: V1–V28 + Scaled Amount + Scaled Time

# Button to load sample
if st.button("🔁 Load Sample Transaction"):
    inputs = [
        -1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396,
        0.0987, 0.3638, 0.0908, -0.5516, -0.6178, -0.9914, -0.3111,
        1.4681, -0.4704, 0.2076, 0.0257, 0.4039, 0.2514, -0.0183,
        0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210,
        0.2405, 0.0339  # Scaled Amount, Scaled Time
    ]
    st.success("✅ Sample values loaded!")

# --- Input Fields ---
user_inputs = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=inputs[i - 1], format="%.4f")
    user_inputs.append(val)

scaled_amount = st.number_input("Scaled Amount", value=inputs[28], format="%.4f")
scaled_time = st.number_input("Scaled Time", value=inputs[29], format="%.4f")
user_inputs.extend([scaled_amount, scaled_time])

# --- Prediction ---
if st.button("🚨 Predict"):
    input_df = pd.DataFrame([user_inputs], columns=[f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ This transaction is FRAUDULENT.")
    else:
        st.success("✅ This transaction is NOT fraudulent.")
