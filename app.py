import streamlit as st
import pandas as pd
import numpy as np
import pickle


with open("decision_tree_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

st.set_page_config(
    page_title="Decision Tree Customer Purchase Prediction",
    layout="centered"
)

st.title("🛒 Customer Purchase Prediction using Decision Tree")

st.write("Enter customer details to predict purchase behavior.")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

salary = st.number_input(
    "Salary",
    min_value=10000,
    max_value=500000,
    value=50000
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

spending_score = st.number_input(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)

membership_years = st.number_input(
    "Membership Years",
    min_value=0,
    max_value=30,
    value=5
)

previous_purchases = st.number_input(
    "Previous Purchases",
    min_value=0,
    max_value=100,
    value=10
)

if st.button("Predict Purchase"):

    gender_encoded = encoder.transform([gender])[0]

    input_data = np.array([[
        age,
        salary,
        credit_score,
        gender_encoded,
        spending_score,
        membership_years,
        previous_purchases
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is Likely to Purchase")
    else:
        st.error("Customer is Not Likely to Purchase")

    st.write(f"Purchase Probability: {probability[1]:.2f}")
    st.write(f"Non-Purchase Probability: {probability[0]:.2f}")

st.markdown("---")
