import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Load model and assets
model = joblib.load("notebook/models/salary_model.pkl")
scaler = joblib.load("notebook/models/scaler.pkl")
feature_columns = joblib.load("notebook/models/feature_columns.pkl")

# Page config
st.set_page_config(page_title="Employee Salary Prediction App", layout="wide", page_icon="💼")

# Style
st.markdown("""
    <style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .title { text-align: center; color: #222; }
    .subtitle { text-align: center; font-size: 1.1rem; color: #777; }
    .center { display: flex; justify-content: center; }
    .metric-container { display: flex; justify-content: space-around; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>💼 Employee Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict whether employees earn >50K using demographic and professional attributes.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar or CSV upload
option = st.radio("Choose Input Method", ["🧍 Manual Entry", "📁 Upload CSV"], horizontal=True)

# Map dictionaries
workclass_map = {
    "Private": "Private",
    "Self-employed (Not Inc.)": "Self-emp-not-inc",
    "Self-employed (Inc.)": "Self-emp-inc",
    "Federal Government": "Federal-gov",
    "Local Government": "Local-gov",
    "State Government": "State-gov",
    "Unpaid": "Without-pay",
    "Never Worked": "Never-worked",
    "Other": "Others"
}

education_map = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS Graduate": 9, "Some College": 10, "Associate (Vocational)": 11,
    "Associate (Academic)": 12, "Bachelors": 13, "Masters": 14,
    "Professional School": 15, "Doctorate": 16
}

marital_map = {
    "Married (Civil)": "Married-civ-spouse",
    "Divorced": "Divorced",
    "Never Married": "Never-married",
    "Separated": "Separated",
    "Widowed": "Widowed",
    "Married (Spouse Absent)": "Married-spouse-absent"
}

occupation_map = {
    "Tech Support": "Tech-support", "Craft/Repair": "Craft-repair",
    "Other Service": "Other-service", "Sales": "Sales",
    "Executive/Managerial": "Exec-managerial", "Professional Specialty": "Prof-specialty",
    "Cleaners": "Handlers-cleaners", "Machine Inspector": "Machine-op-inspct",
    "Clerical": "Adm-clerical", "Farming/Fishing": "Farming-fishing",
    "Transport": "Transport-moving", "Housekeeping": "Priv-house-serv",
    "Protective Services": "Protective-serv", "Military": "Armed-Forces",
    "Others": "Others"
}

relationship_map = {
    "Wife": "Wife", "Child/Dependent": "Own-child",
    "Husband": "Husband", "Not in Family": "Not-in-family",
    "Relative": "Other-relative", "Unmarried": "Unmarried"
}

race_map = {
    "White": "White", "Asian / Pacific Islander": "Asian-Pac-Islander",
    "Native American / Eskimo": "Amer-Indian-Eskimo", "Other": "Other", "Black": "Black"
}

country_map = {
    "United States": "United-States", "Mexico": "Mexico",
    "Philippines": "Philippines", "Germany": "Germany",
    "Canada": "Canada", "India": "India", "England": "England", "Other": "Others"
}

# Manual input
if option == "🧍 Manual Entry":
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 80, 30)
            workclass_ui = st.selectbox("Workclass", list(workclass_map.keys()))
            marital_ui = st.selectbox("Marital Status", list(marital_map.keys()))
            occupation_ui = st.selectbox("Occupation", list(occupation_map.keys()))
            relationship_ui = st.selectbox("Relationship", list(relationship_map.keys()))
        with col2:
            education_ui = st.selectbox("Education Level", list(education_map.keys()))
            race_ui = st.selectbox("Race", list(race_map.keys()))
            gender = st.radio("Gender", ['Male', 'Female'])
            capital_gain = st.number_input("Capital Gain", 0)
            capital_loss = st.number_input("Capital Loss", 0)
            hours_per_week = st.slider("Hours per Week", 1, 99, 40)
            country_ui = st.selectbox("Native Country", list(country_map.keys()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        df_input = pd.DataFrame([{
            'age': age,
            'workclass': workclass_map[workclass_ui],
            'educational-num': education_map[education_ui],
            'marital-status': marital_map[marital_ui],
            'occupation': occupation_map[occupation_ui],
            'relationship': relationship_map[relationship_ui],
            'race': race_map[race_ui],
            'gender': gender,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': country_map[country_ui]
        }])

        encoded = pd.get_dummies(df_input)
        for col in feature_columns:
            if col not in encoded:
                encoded[col] = 0
        encoded = encoded[feature_columns]
        scaled = scaler.transform(encoded)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]*100

        st.subheader("🎯 Prediction Result")
        st.metric("Prediction", ">50K" if pred == 1 else "<=50K")
        st.metric("Confidence", f"{prob:.2f}%")

        st.subheader("📊 Feature Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        chart_data = df_input[['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']].T
        chart_data.columns = ['Value']
        sns.barplot(data=chart_data.reset_index(), x='Value', y='index', palette="coolwarm", ax=ax)
        st.pyplot(fig)

# CSV Upload
elif option == "📁 Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("📋 Uploaded Data Preview", df.head())

        encoded = pd.get_dummies(df)
        for col in feature_columns:
            if col not in encoded:
                encoded[col] = 0
        encoded = encoded[feature_columns]
        scaled = scaler.transform(encoded)

        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1] * 100

        df['Prediction'] = [">50K" if p == 1 else "<=50K" for p in preds]
        df['Confidence (%)'] = probs.round(2)

        st.subheader("🔍 Prediction Summary")
        st.dataframe(df[['Prediction', 'Confidence (%)']].head())

        # Grouped pie chart
        pie_data = df['Prediction'].value_counts()
        st.subheader("📊 Prediction Distribution")
        fig2, ax2 = plt.subplots()
        ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=['#34c759','#ff3b30'])
        ax2.axis('equal')
        st.pyplot(fig2)

        # CSV Download
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        st.download_button("📥 Download Full Results", buffer.getvalue(), "salary_predictions.csv", "text/csv")
