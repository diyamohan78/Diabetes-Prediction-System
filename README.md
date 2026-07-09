# 🩺 Diabetes Prediction System

A Machine Learning-based web application that predicts whether a patient is likely to have diabetes using patient health parameters. The application is developed using **Python**, **Streamlit**, and **Logistic Regression**, providing an intuitive interface for prediction, data analysis, and model performance evaluation.

---

## 📌 Problem Statement

Diabetes is one of the most common chronic diseases worldwide. Early prediction can help healthcare professionals and patients take preventive measures before complications arise.

This project aims to build an intelligent prediction system that estimates the likelihood of diabetes based on medical attributes using Machine Learning.

---

## ✨ Features

- 🔹 Predict diabetes risk using patient health information
- 🔹 Interactive Streamlit web interface
- 🔹 Data Analysis and Visualization
- 🔹 Model Performance Evaluation
- 🔹 Patient Record Management using SQLite
- 🔹 Responsive and user-friendly dashboard

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- SQLite
- Joblib

---

## 🧠 Machine Learning Model

- **Algorithm:** Logistic Regression
- **Problem Type:** Binary Classification
- **Dataset:** Pima Indians Diabetes Dataset
- **Target Variable:** Outcome

---

## 📊 Input Features

The prediction model uses the following medical parameters:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## 📂 Project Structure

```text
Diabetes-Prediction-System/
│
├── assets/
├── images/
├── Pages/
├── app.py
├── train_model.py
├── db_utils.py
├── diabetes.csv
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/coderdiya/Diabetes-Prediction-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📈 Future Enhancements

- Compare multiple Machine Learning models
- Deploy on Streamlit Community Cloud
- User Authentication
- Explainable AI (SHAP)
- Cloud Database Integration

---

## 👩‍💻 Author

**Diya Mohan**

Aspiring Data Analyst | Data Scientist

GitHub: https://github.com/coderdiya

---

⭐ If you found this project useful, consider giving it a star.