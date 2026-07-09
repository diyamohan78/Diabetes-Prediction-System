# ==========================================================
# Diabetes Prediction System
# Model Performance Page
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# Title
# ==========================================================

st.title("📈 Model Performance")

st.write("Performance evaluation of the Logistic Regression model.")

# ==========================================================
# Load Dataset
# ==========================================================

dataset = pd.read_csv("diabetes.csv")

X = dataset.drop("Outcome", axis=1)
y = dataset["Outcome"]

# ==========================================================
# Load Scaler
# ==========================================================

scaler = joblib.load("scaler.pkl")

X_scaled = scaler.transform(X)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Load Model
# ==========================================================

model = joblib.load("model.pkl")

# ==========================================================
# Predictions
# ==========================================================

train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# ==========================================================
# Accuracy
# ==========================================================

train_accuracy = accuracy_score(y_train, train_prediction)
test_accuracy = accuracy_score(y_test, test_prediction)

st.header("🎯 Accuracy")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Accuracy",
        f"{train_accuracy*100:.2f}%"
    )

with col2:
    st.metric(
        "Testing Accuracy",
        f"{test_accuracy*100:.2f}%"
    )

# ==========================================================
# Confusion Matrix
# ==========================================================

st.header("📊 Confusion Matrix")

cm = confusion_matrix(y_test, test_prediction)

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ==========================================================
# Classification Report
# ==========================================================

st.header("📋 Classification Report")

report = classification_report(
    y_test,
    test_prediction,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ==========================================================
# ROC Curve
# ==========================================================

st.header("📈 ROC Curve")

probability = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(
    y_test,
    probability
)

auc = roc_auc_score(
    y_test,
    probability
)

fig, ax = plt.subplots(figsize=(6,5))

ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

ax.plot([0,1],[0,1],"--")

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title("ROC Curve")

ax.legend()

st.pyplot(fig)

# ==========================================================
# Conclusion
# ==========================================================

st.header("📝 Conclusion")

st.success(
    f"""
    ✔ Training Accuracy : {train_accuracy*100:.2f}%

    ✔ Testing Accuracy : {test_accuracy*100:.2f}%

    ✔ ROC-AUC Score : {auc:.3f}

    The Logistic Regression model performs well
    and is suitable for predicting diabetes.
    """
)