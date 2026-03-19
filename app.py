import os
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Titanic Survival Prediction")

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("🚢 Titanic Survival Prediction (RF Model)")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 80, 30)
sibsp = st.sidebar.slider("Siblings/Spouses", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

# -----------------------------
# Encoding
# -----------------------------
sex_enc = 1 if sex == "Male" else 0
embarked_enc = {"C": 0, "Q": 1, "S": 2}[embarked]

input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_enc],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_enc]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # Explicit class-probability mapping (SAFE)
    class_prob = dict(zip(model.classes_, proba))
    confidence = class_prob[pred]

    if pred == 1:
        st.success(f"🟢 Survived (Confidence: {confidence:.2f})")
    else:
        st.error(f"🔴 Not Survived (Confidence: {confidence:.2f})")

    # -----------------------------
    # Display entered details
    # -----------------------------
    st.subheader("Passenger Details")
    st.write(f"Passenger Class: {pclass}")
    st.write(f"Sex: {sex}")
    st.write(f"Age: {age}")
    st.write(f"Siblings/Spouses: {sibsp}")
    st.write(f"Parents/Children: {parch}")
    st.write(f"Fare: {fare}")
    st.write(f"Embarked: {embarked}")

    # -----------------------------
    # Debug (optional)
    # -----------------------------
    with st.expander("🔍 Debug Info"):
        st.write("Model classes:", model.classes_)
        st.write("Probabilities [class-wise]:", class_prob)
