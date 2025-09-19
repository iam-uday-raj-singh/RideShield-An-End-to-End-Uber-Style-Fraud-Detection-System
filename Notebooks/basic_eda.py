import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
matplotlib.use("TkAgg")

# Load dataset
df = pd.read_csv("/Users/udayrajsingh/Desktop/Projects/Uber Fraud Detection/Data/raw/raw_trip_data.csv")

# 1. Distance vs Fare
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="distance_km", y="fare_amount", hue="fraud", alpha=0.6)
plt.title("Distance vs Fare (Fraud vs Legit)")
plt.show(block=True)

# 2. Fraud distribution
sns.countplot(data=df, x="fraud")
plt.title("Fraud vs Legit Trips")
plt.show(block=True)

# 3. Device Reuse
device_counts = df["device_id"].value_counts().head(10)
device_counts.plot(kind="bar", title="Top Device Usage (Potential Fraud)")
plt.show(block=True)

# 4. Rider–Driver Repeat Check
pairs = df.groupby(["rider_id", "driver_id"]).size().reset_index(name="count")
fraud_pairs = pairs[pairs["count"] > 5]
print("🚩 Suspicious Rider-Driver Pairs:")
print(fraud_pairs.head())
