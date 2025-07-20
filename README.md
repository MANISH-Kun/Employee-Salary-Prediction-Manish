# 💼 Employee Salary Prediction – Manish Kun

This project predicts whether an individual earns **more than $50K per year** based on demographic and professional attributes using Machine Learning. Built as part of the **AICTE – Edunet Foundation IBM SkillsBuild Internship**, it includes a web interface powered by **Streamlit** for both manual and batch predictions.

---

## 🧠 Problem Statement

The objective is to classify employees into two salary classes:

- **`<=50K`**
- **`>50K`**

using a dataset containing attributes such as age, education, occupation, working hours, and more. This helps in understanding patterns behind salary distribution and can support data-driven HR decisions.

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


---

## 🚀 How to Run the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/MANISH-Kun/Employee-Salary-Prediction-Manish.git
cd Employee-Salary-Prediction-Manish

### 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows

# 3. Install dependencies
pip install -r requirements.txt

#4. Run the Streamlit App
streamlit run app.py

#The app will launch in your browser at http://localhost:8501.
