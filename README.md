# 💼 Employee Salary Prediction – Manish Kun

This project predicts whether an individual earns more than $50K per year based on demographic and professional attributes using machine learning techniques. It was developed as part of the **AICTE – Edunet Foundation IBM SkillsBuild Internship** program.

---

## 🧠 Problem Statement

The objective is to create a predictive model that classifies individuals into two income groups:
- `<=50K`
- `>50K`

Given a dataset with features like age, education, occupation, and working hours, the goal is to apply various machine learning algorithms and deploy the best one using a user-friendly web interface.

---

## ✅ Features

- 🔍 Predict income class using manual input or CSV batch upload
- 📊 Visualize feature importance and prediction confidence
- 📈 Pie charts and bar charts for interpretability
- 💻 Streamlit-powered frontend
- 📥 Download prediction results as CSV
- 🧪 Trained using multiple algorithms (best: **Gradient Boosting**)

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** – Data handling
- **Scikit-learn, XGBoost** – Machine Learning
- **Matplotlib, Seaborn** – Visualization
- **Streamlit** – Web application framework
- **Joblib** – Model persistence

---

## 🧪 Models Tested

| Algorithm             | Accuracy (%) |
|-----------------------|--------------|
| Logistic Regression   | 84.7         |
| Decision Tree         | 85.9         |
| KNN                   | 86.1         |
| Random Forest         | 86.8         |
| Gradient Boosting     | ✅ **87.9**  |
| XGBoost               | 87.2         |

---

## 🧾 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MANISH-Kun/Employee-Salary-Prediction-Manish.git
   cd Employee-Salary-Prediction-Manish
