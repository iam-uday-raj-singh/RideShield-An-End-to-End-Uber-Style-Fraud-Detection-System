

# 🚖 RideSheild-An-End-to-End-Uber-Style-Fraud-Detection-System

Fraudulent activity in ride-hailing platforms (like Uber, Ola) can cause **financial loss** and **trust issues** for both customers and drivers.
This project builds an **end-to-end machine learning pipeline** to detect fraudulent trips, with a focus on **explainability** so stakeholders can understand why a trip was flagged.

---

## 📌 Project Overview

* **Problem**: Fraudulent trips such as inflated fares, repeated rider–driver collusion, and device reuse.
* **Solution**: A machine learning pipeline that detects anomalies using engineered trip features.
* **Key Highlight**: Models are explainable via **SHAP values** and **feature importance**, making it easier to trust predictions.

---

## ⚙️ Project Phases

### 1. Data Generation

* Since real Uber trip fraud datasets are not publicly available, a **synthetic dataset** was generated using the `Faker` library.
* Dataset includes:

  * Rider & driver IDs
  * Pickup & dropoff locations
  * Pickup & dropoff times
  * Distance & fare amount
  * Payment method
  * Fraud type (`legit`, `short_high_fare`, `rider_driver_repeat`, `device_reuse`)

### 2. Exploratory Data Analysis (EDA)

* Fraud distribution visualization.
* Distance vs. fare relationship.
* Device reuse analysis.
* Suspicious rider–driver repeated pairs.

### 3. Feature Engineering

* Derived features:

  * `trip_duration_min`, `trip_speed`
  * `fare_distance_ratio`
  * `trips_per_driver_hour`
  * `device_rider_ratio`
  * `geo_distance_consecutive_trips`
* Label encoding + scaling.

### 4. Model Training & Explainability

* Models trained:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Performance: \~99% accuracy (on synthetic dataset).
* **Explainability**:

  * SHAP summary plots
  * Feature importance charts

### 5. Model Optimization (Planned)

* Hyperparameter tuning with GridSearchCV / Optuna.
* Cross-validation to reduce overfitting.
* Deployment pipeline with saved `.pkl` models.

---

## 📊 Tech Stack

* **Language**: Python 3
* **Libraries**:

  * Data: `pandas`, `numpy`, `faker`
  * Visualization: `matplotlib`, `seaborn`
  * ML Models: `scikit-learn`, `xgboost`
  * Explainability: `shap`
  * Utilities: `joblib`

---

## 📂 Project Structure

```
Uber_Fraud_Detection/
│── Data/
│   ├── raw/                 # Synthetic raw dataset
│   ├── processed/           # Processed dataset (feature engineered)
│── Notebooks/               # Jupyter notebooks for EDA
│── SRC/                     # Source code
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│── Models/                  # Saved trained models (.pkl)
│── README.md
```

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/Uber_Fraud_Detection.git
   cd Uber_Fraud_Detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate synthetic dataset:

   ```bash
   python SRC/data_generation.py
   ```

4. Run feature engineering:

   ```bash
   python SRC/feature_engineering.py
   ```

5. Train models & generate explainability plots:

   ```bash
   python SRC/model_training.py
   ```

---

## 📌 Results

* **Models achieve \~99% accuracy** (synthetic dataset).
* Fraud detection scenarios successfully captured:

  * High fare for short trips
  * Repeated rider–driver pairs
  * Device reuse across multiple riders
* SHAP plots highlight **fare-distance ratio** and **trip speed** as top fraud indicators.

---

## 🔮 Future Work

* Replace synthetic data with **real-world datasets** (if available).
* Deploy as a **REST API** (Flask/FastAPI).
* Build a **Streamlit dashboard** for fraud monitoring.
* Expand fraud types with more realistic scenarios.

---
