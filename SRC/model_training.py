# Phase 4: Model Training and Explainability

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# 1. Load processed data
# ------------------------------
df = pd.read_csv("/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Data/processed/processed_trip_data.csv")

print("Data loaded successfully ✅")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------
# 2. Features & Target
# ------------------------------
X = df.drop("fraud_flag", axis=1)
y = df["fraud_flag"]

# Encode categorical features if any
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ------------------------------
# 3. Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 4. Train Models
# ------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)  # RF does not require scaling

# ------------------------------
# 5. Evaluation
# ------------------------------
print("\n📊 Logistic Regression Report")
print(classification_report(y_test, log_reg.predict(X_test_scaled)))

print("\n📊 Random Forest Report")
print(classification_report(y_test, rf.predict(X_test)))

# ------------------------------
# 6. Save Models
# ------------------------------
save_dir = "/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Models"
os.makedirs(save_dir, exist_ok=True)

log_path = os.path.join(save_dir, "log_reg_model.pkl")
rf_path = os.path.join(save_dir, "rf_model.pkl")
scaler_path = os.path.join(save_dir, "scaler.pkl")

joblib.dump(log_reg, log_path)
joblib.dump(rf, rf_path)
joblib.dump(scaler, scaler_path)

print("\n✅ Models saved at:")
print(" - Logistic Regression:", log_path, "| Exists?", os.path.exists(log_path))
print(" - Random Forest:", rf_path, "| Exists?", os.path.exists(rf_path))
print(" - Scaler:", scaler_path, "| Exists?", os.path.exists(scaler_path))

# ------------------------------
# 7. Explainability with SHAP
# ------------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Summary plot (feature importance)
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance")
plt.show()
