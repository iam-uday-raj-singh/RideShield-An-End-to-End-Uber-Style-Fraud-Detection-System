import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker()

# Function to generate random coordinates near New Delhi
def random_location():
    lat = 28.61 + random.uniform(-0.1, 0.1)  # near New Delhi lat
    lon = 77.23 + random.uniform(-0.1, 0.1)  # near New Delhi lon
    return lat, lon

# Generate dataset
records = []
for i in range(11000):  # 10k legit + ~1k fraud
    trip_id = i + 1
    rider_id = fake.uuid4()
    driver_id = fake.uuid4()
    device_id = random.choice([fake.uuid4() for _ in range(8000)])  # device reuse possible

    start_lat, start_lon = random_location()
    end_lat, end_lon = random_location()

    # pickup and dropoff time
    pickup_time = fake.date_time_between(start_date="-90d", end_date="now")
    dropoff_time = pickup_time + pd.to_timedelta(random.randint(5, 60), unit='m')

    # calculate distance and fare
    distance_km = round(np.sqrt((end_lat - start_lat)**2 + (end_lon - start_lon)**2) * 100, 2)
    base_fare = distance_km * random.uniform(8, 12)  

    fraud_flag = 0
    fare_amount = round(base_fare, 2)

    # Fraud injection (~9%)
    if i % 11 == 0:  
        fraud_flag = 1
        fraud_type = random.choice(["short_high_fare", "rider_driver_repeat", "device_reuse"])
        
        if fraud_type == "short_high_fare":
            distance_km = round(random.uniform(0.1, 0.5), 2)
            fare_amount = round(random.uniform(300, 800), 2)
        
        elif fraud_type == "rider_driver_repeat":
            rider_id = "fraud_rider_123"
            driver_id = "fraud_driver_456"
        
        elif fraud_type == "device_reuse":
            device_id = "shared_device_999"
    else:
        fraud_type = "legit"
    
    fraud_flag = 0
    fraud_type = "legit"

    if i % 11 == 0:
        fraud_flag = 1
        fraud_type = random.choice(["short_high_fare", "rider_driver_repeat", "device_reuse"])


    records.append([
        trip_id, driver_id, rider_id, device_id,
        f"{start_lat},{start_lon}", f"{end_lat},{end_lon}", distance_km,
        fare_amount, random.choice(["cash", "card", "wallet"]),
        pickup_time, dropoff_time, fraud_flag, fraud_type
    ])

# Create dataframe
df = pd.DataFrame(records, columns=[
    'trip_id','driver_id','rider_id','device_id',
    'pickup_location','dropoff_location','distance_km',
    'fare_amount','payment_method','pickup_time','dropoff_time','fraud_flag','fraud_type'
])

# Save dataset
df.to_csv("/Users/udayrajsingh/Desktop/Projects/Uber Fraud Detection/Data/raw/raw_trip_data.csv", index=False)
print("✅ Dataset updated with pickup & dropoff time and saved.")
