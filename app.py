import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load(open('artefacts/loan_prediction_model.pkl', 'rb'))
scaler =joblib.load(open('artefacts/scaler.pkl', 'rb'))

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details below to predict loan approval status:")

st.sidebar.header("About Project")
st.sidebar.info("This Loan Prediction app was developed as part of the NIELET Data Science with Python Training.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

input_data = pd.DataFrame({
    'Gender': [1 if gender == 'Male' else 0],
    'Married': [1 if married == 'Yes' else 0],
    'Dependents': [dependents],
    'Education': [1 if education == 'Graduate' else 0],
    'Self_Employed': [1 if self_employed == 'Yes' else 0],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [0 if property_area == 'Rural' else (1 if property_area == 'Semiurban' else 2)],
    'Total_Income': [applicant_income + coapplicant_income],
    'Total_Income_log': [np.log(applicant_income + coapplicant_income + 1)],
    'LoanAmount_log': [np.log(loan_amount + 1)],
    'Loan_Amount_Term_Years': [loan_term / 12.0],
    'Income_to_Loan': [(applicant_income + coapplicant_income) / (loan_amount + 1)],
    'Income_to_Loan_log': [np.log((applicant_income + coapplicant_income) / (loan_amount + 1) + 1)]
})

# Scale the data
scaled_input = scaler.transform(input_data)

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")