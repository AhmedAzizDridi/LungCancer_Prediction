# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

st.set_page_config(page_title="Lung Cancer Survival Prediction", page_icon="🫁", layout="centered")

st.title("🫁 Lung Cancer Survival Prediction")
st.write("Enter the details below to estimate the patient's survival probability.")


@st.cache_resource
def load_artifacts():
    model = load_model("model1.h5")
    scaler = joblib.load("scaler1.pkl")   # scaler trained on the same 18 columns/order below
    return model, scaler

model, scaler = load_artifacts()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=24.0, step=0.1)
    cholesterol_level = st.number_input("Cholesterol Level", min_value=0.0, max_value=500.0, value=180.0, step=1.0)

    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    asthma = st.selectbox("Asthma", ["No", "Yes"])
    cirrhosis = st.selectbox("Cirrhosis", ["No", "Yes"])
    other_cancer = st.selectbox("Other Cancer", ["No", "Yes"])

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
    family_history = st.selectbox("Family History", ["No", "Yes"])
    smoking_status = st.selectbox(
        "Smoking Status",
        ["Current Smoker", "Former Smoker", "Never Smoked", "Passive Smoker"]
    )
    treatment_type = st.selectbox(
        "Treatment Type",
        ["Chemotherapy", "Radiation", "Surgery", "Combined"]
    )

st.markdown("---")

# Fixed decision threshold picked from your validation tuning
THRESHOLD = 0.6588  # predict SURVIVE when P(survive) >= 0.6588

# -----------------------------
# Build features in EXACT order (18 columns)
# Only age, bmi, cholesterol_level are numeric; others are 0/1.
# Baselines: Female, Stage I, Current Smoker, Chemotherapy
# -----------------------------
FEATURE_ORDER = [
    "age",
    "bmi",
    "cholesterol_level",
    "hypertension",
    "asthma",
    "cirrhosis",
    "other_cancer",
    "gender_Male",
    "cancer_stage_Stage II",
    "cancer_stage_Stage III",
    "cancer_stage_Stage IV",
    "family_history_Yes",
    "smoking_status_Former Smoker",
    "smoking_status_Never Smoked",
    "smoking_status_Passive Smoker",
    "treatment_type_Combined",
    "treatment_type_Radiation",
    "treatment_type_Surgery",
]

def yesno(s: str) -> int:
    return 1 if s == "Yes" else 0

row = {col: 0 for col in FEATURE_ORDER}
# numeric
row["age"] = float(age)
row["bmi"] = float(bmi)
row["cholesterol_level"] = float(cholesterol_level)
# binaries
row["hypertension"] = yesno(hypertension)
row["asthma"] = yesno(asthma)
row["cirrhosis"] = yesno(cirrhosis)
row["other_cancer"] = yesno(other_cancer)
row["family_history_Yes"] = yesno(family_history)
# one-hots from selections (baseline left at 0)
if gender == "Male":
    row["gender_Male"] = 1

if cancer_stage == "Stage II":
    row["cancer_stage_Stage II"] = 1
elif cancer_stage == "Stage III":
    row["cancer_stage_Stage III"] = 1
elif cancer_stage == "Stage IV":
    row["cancer_stage_Stage IV"] = 1
# Stage I -> baseline

if smoking_status == "Former Smoker":
    row["smoking_status_Former Smoker"] = 1
elif smoking_status == "Never Smoked":
    row["smoking_status_Never Smoked"] = 1
elif smoking_status == "Passive Smoker":
    row["smoking_status_Passive Smoker"] = 1
# Current Smoker -> baseline

if treatment_type == "Combined":
    row["treatment_type_Combined"] = 1
elif treatment_type == "Radiation":
    row["treatment_type_Radiation"] = 1
elif treatment_type == "Surgery":
    row["treatment_type_Surgery"] = 1
# Chemotherapy -> baseline

X_input = pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

# -----------------------------
# Predict (model outputs P(dead) since 1 = dead)
# -----------------------------
C_FP, C_FN = 1, 4
thr = C_FP / (C_FP + C_FN)

if st.button("Predict"):
    X_scaled = scaler.transform(X_input)
    p_dead = float(model.predict(X_scaled, verbose=0).ravel()[0])

    pred_class = 1 if p_dead >= thr else 0
    labels = {0: "live", 1: "die"}

  
    st.success(f"**The patient is likely to:** {labels[pred_class]}")
    



