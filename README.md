# 💼 Employee Salary Prediction – Manish 

This project predicts whether an individual earns **more than $50K per year** based on demographic and professional attributes using Machine Learning. Built as part of the **AICTE – Edunet Foundation IBM SkillsBuild Internship**, it includes a web interface powered by **Streamlit** for both manual and batch predictions.

---

## 🧠 Problem Statement

In today’s data-driven world, organizations and governments face challenges in identifying individuals with high earning potential based on limited demographic and professional information. There is a growing need for intelligent systems that can analyze factors like education, age, work hours, occupation, and more, to predict income brackets accurately. This project aims to build a machine learning model that predicts whether an individual earns more than $50,000 annually. By solving this problem, companies can make better hiring decisions, governments can understand economic patterns, and data science learners can gain practical experience. The motive behind choosing this project is to address real-world socio-economic challenges while applying predictive analytics and deploying the model through an interactive Streamlit web application.


---

## ✅ Features

- 🔍 Manual entry prediction form
- 📁 Batch prediction via CSV upload
- 📈 Interactive visualizations (bar charts, pie charts)
- 🎯 Confidence score with every prediction
- 📥 Downloadable prediction CSV
- 💻 Clean, modern Streamlit UI
- 🧠 Trained on multiple ML algorithms – best selected (Gradient Boosting)

---

## 🛠️ Tech Stack

| Area              | Tools Used                               |
|-------------------|------------------------------------------|
| Programming       | Python                                   |
| ML Libraries      | Scikit-learn, XGBoost                    |
| Data Handling     | Pandas, NumPy                            |
| Visualization     | Seaborn, Matplotlib                      |
| Model Saving      | Joblib                                   |
| Web UI            | Streamlit                                |

---

## 🧪 ML Models Tested

| Algorithm             | Accuracy (%) |
|-----------------------|--------------|
| Logistic Regression   | 84.7         |
| Decision Tree         | 85.9         |
| KNN                   | 86.1         |
| Random Forest         | 86.8         |
| Gradient Boosting     | ✅ **87.9**  |
| XGBoost               | 87.2         |

✅ **Gradient Boosting** was selected as the final model for deployment.

---
watch here:  https://employee-salary-prediction-manish-zeno.streamlit.app/


## 📁 Project Structure
```
employee_salary_prediction/
│
├── data/
│   └── adult.csv
│
├── notebook/
│   ├── salary_prediction.ipynb
│   └── models/
│       ├── salary_model.pkl
│       ├── scaler.pkl
│       └── feature_columns.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/MANISH-Kun/Employee-Salary-Prediction-Manish.git
cd Employee-Salary-Prediction-Manish

### 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Linux or macOS

# 3. Install dependencies
pip install -r requirements.txt

#4. Run the Streamlit App
streamlit run app.py

#The app will launch in your browser at http://localhost:8501.
