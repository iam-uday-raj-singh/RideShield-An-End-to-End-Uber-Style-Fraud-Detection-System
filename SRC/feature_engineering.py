import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load raw dataset
df = pd.read_csv("/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Data/raw/raw_trip_data.csv")

# -----------------------------
# Feature Engineering
# -----------------------------

# 1. Trip duration (minutes)
df['trip_duration_min'] = (pd.to_datetime(df['dropoff_time']) - pd.to_datetime(df['pickup_time'])).dt.total_seconds() / 60

# 2. Trip speed (km/h) = distance / time
df['trip_speed'] = df['distance_km'] / (df['trip_duration_min'] / 60 + 1e-5)

# 3. Fare per km (fare-distance ratio)
df['fare_distance_ratio'] = df['fare_amount'] / (df['distance_km'] + 1e-5)

# 4. Trips per driver per hour (group-level)
df['pickup_time'] = pd.to_datetime(df['pickup_time'])
df['driver_hour'] = df['driver_id'].astype(str) + "_" + df['pickup_time'].dt.strftime('%Y-%m-%d %H')
driver_counts = df.groupby('driver_hour').size().reset_index(name='trips_per_driver_hour')
df = df.merge(driver_counts, on='driver_hour', how='left')

# 5. Device-rider ratio = number of riders per device
device_rider_counts = df.groupby('device_id')['rider_id'].nunique().reset_index(name='device_rider_ratio')
df = df.merge(device_rider_counts, on='device_id', how='left')

# 6. Geo distance between consecutive trips by same driver
df = df.sort_values(by=['driver_id', 'pickup_time'])
df['prev_drop_lat'] = df['dropoff_location'].shift().str.split(",").str[0].astype(float)
df['prev_drop_lon'] = df['dropoff_location'].shift().str.split(",").str[1].astype(float)
df['curr_pick_lat'] = df['pickup_location'].str.split(",").str[0].astype(float)
df['curr_pick_lon'] = df['pickup_location'].str.split(",").str[1].astype(float)

df['geo_distance_consecutive_trips'] = np.where(
    df['driver_id'] == df['driver_id'].shift(),
    np.sqrt((df['curr_pick_lat'] - df['prev_drop_lat'])**2 + (df['curr_pick_lon'] - df['prev_drop_lon'])**2) * 100,
    0
)

# -----------------------------
# Encode categorical features
# -----------------------------
label_enc = LabelEncoder()

# Encode payment method
if 'payment_method' in df.columns:
    df['payment_method_encoded'] = label_enc.fit_transform(df['payment_method'])

# Encode fraud type if available
if 'fraud_type' in df.columns:
    print("🔎 Unique fraud_type categories before encoding:", df['fraud_type'].unique())
    df['fraud_type_encoded'] = label_enc.fit_transform(df['fraud_type'])
    print("✅ fraud_type encoded successfully.")
else:
    print("⚠️ fraud_type column not found — skipping encoding.")

# -----------------------------
# Handle missing values
# -----------------------------
df.fillna(0, inplace=True)

# -----------------------------
# Scaling numeric features
# -----------------------------
scaler = StandardScaler()
scale_cols = ['trip_duration_min', 'trip_speed', 'fare_distance_ratio', 
              'trips_per_driver_hour', 'device_rider_ratio', 'geo_distance_consecutive_trips']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Encode fraud type if available
if 'fraud_type' in df.columns:
    print("🔎 Unique fraud_type categories before encoding:", df['fraud_type'].unique())
    df['fraud_type_encoded'] = label_enc.fit_transform(df['fraud_type'])
    
    # Create mapping dictionary
    fraud_mapping = {label: idx for idx, label in enumerate(label_enc.classes_)}
    print("✅ fraud_type encoding mapping:", fraud_mapping)
else:
    print("⚠️ fraud_type column not found — skipping encoding.")


# -----------------------------
# Save processed dataset
# -----------------------------
df.to_csv("/Users/udayrajsingh/Desktop/Projects/Uber_Fraud_Detection/Data/processed/processed_trip_data.csv", index=False)
print("✅ Feature engineering complete! Processed dataset saved at data/processed/processed_trip_data.csv")
