# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("notebook/salary_model.pkl")

# Load encoder and column names
encoder = joblib.load("notebook/models/encoder.pkl")
cat_cols = joblib.load("notebook/models/categorical_columns.pkl")
num_cols = joblib.load("notebook/models/numerical_columns.pkl")

# Page title
st.set_page_config(page_title="Salary Class Predictor")
st.title("💼 Employee Salary Income Predictor")
st.markdown("Predict whether income is **>50K** or **<=50K** based on employee details")

# Input form
def user_input():
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Others'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Others'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England', 'Others'])

    # Construct DataFrame
    data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    return data

# Get input
user_data = user_input()

# Split into categorical and numerical
user_cat = user_data[cat_cols]
user_num = user_data[num_cols]

# Encode categorical
encoded_cat = encoder.transform(user_cat)

# Combine all
final_input = np.concatenate([encoded_cat, user_num.values], axis=1)

# Predict
prediction = model.predict(final_input)
prediction_label = ">50K" if prediction[0] == 1 else "<=50K"

# Output
st.subheader("Prediction:")
st.success(f"💰 Income is likely **{prediction_label}**")
