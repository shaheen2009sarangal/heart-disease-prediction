import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Feature names
feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                 'Oldpeak', 'ST_Slope']

# App Title
st.title("💓 Heart Disease Prediction System")
st.write("Enter patient details below to predict heart disease risk.")

# Input Fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox("Sex", options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male")

cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
    format_func=lambda x: {0:"ASY — Asymptomatic", 1:"ATA — Atypical Angina",
                            2:"NAP — Non-Anginal Pain", 3:"TA — Typical Angina"}[x])

restingbp = st.number_input("Resting Blood Pressure (mm Hg)",
    min_value=50, max_value=250, value=120)

chol = st.number_input("Cholesterol (mg/dl)",
    min_value=0, max_value=600, value=200)

fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
    format_func=lambda x: {0:"LVH", 1:"Normal", 2:"ST"}[x])

maxhr = st.number_input("Maximum Heart Rate Achieved",
    min_value=50, max_value=250, value=150)

exang = st.selectbox("Exercise Induced Angina",
    options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

oldpeak = st.number_input("Oldpeak (ST Depression)",
    min_value=-3.0, max_value=10.0, value=0.0)

slope = st.selectbox("ST Slope", options=[0, 1, 2],
    format_func=lambda x: {0:"Down", 1:"Flat", 2:"Up"}[x])

# Predict Button
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, restingbp, chol, fastingbs,
                            restecg, maxhr, exang, oldpeak, slope]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Result:")

    # ── Option 1: Probability ──
    risk_percent = round(probability[1] * 100, 2)
    healthy_percent = round(probability[0] * 100, 2)

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk — Heart Disease Detected")
        st.markdown("""
        <div style='text-align: center; font-size: 120px;'>
            😢❤️‍🩹
        </div>
        <div style='text-align: center; font-size: 24px; color: #e74c3c; font-weight: bold;'>
            Please consult a doctor immediately
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Low Risk — No Heart Disease Detected")
        st.markdown("""
        <div style='text-align: center; font-size: 120px;'>
            😊💚
        </div>
        <div style='text-align: center; font-size: 24px; color: #2ecc71; font-weight: bold;'>
            Your heart looks healthy! Keep it up!
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"### 🎯 Prediction Confidence")
    col1, col2 = st.columns(2)
    col1.metric("Heart Disease Risk", f"{risk_percent}%")
    col2.metric("Healthy Probability", f"{healthy_percent}%")

    fig1, ax1 = plt.subplots(figsize=(6, 1.5))
    ax1.barh(["Risk"], [risk_percent], color='#e74c3c')
    ax1.barh(["Risk"], [100 - risk_percent],
             left=[risk_percent], color='#2ecc71')
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Probability (%)")
    ax1.set_title("Heart Disease Risk Meter")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    st.pyplot(fig1)

    # ── Option 3: Feature Importance ──
    st.markdown("### 📊 Key Factors Influencing This Prediction")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.barh(sorted_features[::-1],
                    sorted_importances[::-1], color='#3498db')
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Feature Importance — What Affects Prediction Most")
    st.pyplot(fig2)

    st.warning("⚠️ Disclaimer: This system is for academic purposes only. "
               "Please consult a qualified doctor for medical advice.")

# ── Accuracy Chart (always visible) ──
st.markdown("---")
st.subheader("📈 Model Accuracy Comparison")
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [84.78, 86.41, 88.04]
colors = ['#3498db', "#cb2179", "#84e013e4"]

fig3, ax3 = plt.subplots()
bars = ax3.bar(models, accuracies, color=colors)
ax3.set_ylim(70, 100)
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Model Accuracy Comparison")

for bar, acc in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{acc}%', ha='center', va='bottom', fontweight='bold')

st.pyplot(fig3)