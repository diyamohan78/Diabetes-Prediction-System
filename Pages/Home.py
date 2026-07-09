import streamlit as st

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Prediction System")

st.markdown("""
Welcome to the **Diabetes Prediction System**, a Machine Learning-powered web application
designed to predict whether a patient is likely to have diabetes based on various health parameters.

This application helps users understand diabetes risk using an easy-to-use interface and provides
interactive data analysis and model performance insights.
""")

st.divider()

st.header("🎯 Project Objective")

st.write("""
The objective of this project is to build a Machine Learning model that predicts diabetes
using patient medical information. Early prediction can help in timely diagnosis and treatment.
""")

st.divider()

st.header("✨ Key Features")

col1, col2 = st.columns(2)

with col1:
    st.success("✔ Diabetes Prediction")
    st.success("✔ Data Analysis Dashboard")
    st.success("✔ Interactive Visualizations")

with col2:
    st.success("✔ Model Performance")
    st.success("✔ Patient Record Storage")
    st.success("✔ User-Friendly Interface")

st.divider()

st.header("⚙️ Technologies Used")

st.markdown("""
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Matplotlib
- Seaborn
- SQLite
""")

st.divider()

st.header("🚀 How to Use")

st.markdown("""
1. Open the **Diabetes Prediction** page.
2. Enter the patient's health information.
3. Click **Predict**.
4. View the prediction result.
5. Explore the **Data Analysis** and **Model Performance** pages.
""")

st.info("💡 This project is developed for educational purposes and demonstrates the use of Machine Learning in healthcare.")