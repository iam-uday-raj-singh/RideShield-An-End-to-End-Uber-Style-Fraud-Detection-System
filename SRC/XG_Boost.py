import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Load processed dataset
# -----------------------------
df = pd.read_csv("/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Data/processed/processed_trip_data.csv")

# -----------------------------
# 2. Encode target variable
# -----------------------------
le_target = LabelEncoder()
df["fraud_type_encoded"] = le_target.fit_transform(df["fraud_type"])

# -----------------------------
# 3. Encode payment_method
# -----------------------------
le_payment = LabelEncoder()
df["payment_method_encoded"] = le_payment.fit_transform(df["payment_method"])

# -----------------------------
# 4. Convert datetime columns to numeric features
# -----------------------------
df['pickup_time'] = pd.to_datetime(df['pickup_time'])
df['dropoff_time'] = pd.to_datetime(df['dropoff_time'])

df['pickup_hour'] = df['pickup_time'].dt.hour
df['pickup_dayofweek'] = df['pickup_time'].dt.dayofweek
df['pickup_day'] = df['pickup_time'].dt.day

df['dropoff_hour'] = df['dropoff_time'].dt.hour
df['dropoff_dayofweek'] = df['dropoff_time'].dt.dayofweek
df['dropoff_day'] = df['dropoff_time'].dt.day

# -----------------------------
# 5. Encode driver_hour for trips per driver per hour
# -----------------------------
df['driver_hour'] = df['driver_id'].astype(str) + "_" + df['pickup_time'].dt.strftime('%Y-%m-%d %H')
le_driver_hour = LabelEncoder()
df['driver_hour_encoded'] = le_driver_hour.fit_transform(df['driver_hour'])

# -----------------------------
# 6. Drop non-numeric/string columns
# -----------------------------
drop_cols = [
    "trip_id", "rider_id", "driver_id", "device_id",
    "pickup_location", "dropoff_location", "fraud_type", "payment_method",
    "pickup_time", "dropoff_time", "driver_hour"
]

X = df.drop(columns=drop_cols + ["fraud_type_encoded"])
y = df["fraud_type_encoded"]

print("✅ Features used for XGBoost:", list(X.columns))

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 8. Scale numeric features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 9. Train XGBoost model
# -----------------------------
xgb_clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(le_target.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)

xgb_clf.fit(X_train_scaled, y_train)

# -----------------------------
# 10. Evaluate
# -----------------------------
y_pred = xgb_clf.predict(X_test_scaled)
print("\n📊 XGBoost Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 11. Save model, scaler, encoders
# -----------------------------
joblib.dump(xgb_clf, "/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Models/xgboost_model.pkl")
joblib.dump(scaler, "/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Models/scaler.pkl")
joblib.dump(le_target, "/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Models/label_encoder_target.pkl")
joblib.dump(le_payment, "/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Models/label_encoder_payment.pkl")
joblib.dump(le_driver_hour, "/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Models/label_encoder_driver_hour.pkl")
print("\n✅ XGBoost model, scaler, and encoders saved successfully!")

# -----------------------------
# 12. SHAP Explainability
# -----------------------------
explainer = shap.Explainer(xgb_clf, X_train_scaled)
shap_values = explainer(X_test_scaled[:100])

plt.title("SHAP Feature Importance (XGBoost)")
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.show()

# -----------------------------
# 13. Per-class SHAP Explainability
# -----------------------------
# SHAP values for all test data
explainer = shap.Explainer(xgb_clf, X_train_scaled)
shap_values_all = explainer(X_test_scaled)

# Get class names
class_names = le_target.classes_

for i, class_name in enumerate(class_names):
    print(f"\n🌟 SHAP Summary Plot for class: {class_name}")
    # shap_values_all.values has shape (num_samples, num_classes, num_features)
    shap_values_class = shap_values_all.values[:, i, :]
    shap.summary_plot(
        shap_values_class,
        X_test,
        feature_names=X.columns,
        show=True
    )
