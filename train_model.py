# ==========================================================
# Diabetes Prediction System
# Train and Save Logistic Regression Model
# ==========================================================

# ---------------------- Import Libraries ----------------------

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================
# Load Dataset
# ==========================================================

dataset = pd.read_csv("diabetes.csv")

# ==========================================================
# Features and Target
# ==========================================================

X = dataset.drop("Outcome", axis=1)
y = dataset["Outcome"]

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

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
# Train Logistic Regression Model
# ==========================================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

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

print("=" * 50)
print("Training Accuracy :", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy  :", round(test_accuracy * 100, 2), "%")
print("=" * 50)

# ==========================================================
# Confusion Matrix
# ==========================================================

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, test_prediction))

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")
print(classification_report(y_test, test_prediction))

# ==========================================================
# Save Model and Scaler
# ==========================================================

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model Saved Successfully!")
print("Generated Files:")
print("✔ model.pkl")
print("✔ scaler.pkl")