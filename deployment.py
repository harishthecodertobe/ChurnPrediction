import streamlit as st
import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort

# -----------------------------
# Load encoders & scaler
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

# -----------------------------
# Load ONNX model
# -----------------------------
session = ort.InferenceSession("model.onnx")

# -----------------------------
# UI
# -----------------------------
st.title("Customer Churn Prediction (ONNX Model)")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 30)
credit_score = st.slider("Credit Score", 300, 900, 600)
balance = st.slider("Balance", 0, 2500000)
estimated_salary = st.slider("Estimated Salary", 0, 200000)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])

# -----------------------------
# Prepare input data
# -----------------------------
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# Geography → One-hot
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Merge
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale
input_scaled = scaler.transform(input_data).astype(np.float32)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Churn"):
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: input_scaled})[0][0][0]

    if pred > 0.5:
        st.error(f"Customer is likely to churn with probability {pred:.2f}")
    else:
        st.success(f"Customer is unlikely to churn with probability {1 - pred:.2f}")

    st.progress(float(pred))
