"""
Synthetic UPI Transaction Dataset Generator (10,000 Records)
Generates realistic Indian Unified Payments Interface (UPI) transaction data
with realistic fraud typologies (Impossible Travel, Velocity Drain, Dormant Midnight Spikes,
Phishing QR Collect Requests, Mule Accounts, and Device Fingerprint Mismatches).
"""

import random
import time
import math
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Major Indian Cities & International Coordinates
LOCATIONS = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab"},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "London_Anomaly": {"lat": 51.5074, "lon": -0.1278, "state": "UK"},
    "Lagos_Anomaly": {"lat": 6.5244, "lon": 3.3792, "state": "Nigeria"},
    "Moscow_Anomaly": {"lat": 55.7558, "lon": 37.6173, "state": "Russia"}
}

INDIAN_CITIES = [k for k in LOCATIONS.keys() if "_Anomaly" not in k]

# Popular UPI Handles & Banks
UPI_HANDLES = ["@oksbi", "@okhdfcbank", "@okicici", "@okaxis", "@paytm", "@ybl", "@ibl", "@apl", "@barodampay"]
MERCHANT_HANDLES = ["@hdfcbank", "@icici", "@paytmqr", "@bharatpe", "@razorpay", "@sbi"]

FIRST_NAMES = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohan", "Pooja", "Arjun", "Kavita",
               "Sanjay", "Deepika", "Aditya", "Neha", "Manish", "Divya", "Karan", "Tanvi", "Naveen", "Shreya",
               "Rajesh", "Meera", "Varun", "Sunita", "Harsh", "Ritika", "Gaurav", "Simran", "Nikhil", "Akanksha"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Iyer", "Reddy", "Nair", "Mehta",
              "Chopra", "Joshi", "Bose", "Kulkarni", "Deshmukh", "Aggarwal", "Rao", "Mishra", "Pandey", "Saxena"]

MERCHANTS = [
    ("Swiggy Delivery", "swiggy.orders", "P2M", 150, 900),
    ("Zomato Online", "zomato.pay", "P2M", 200, 1200),
    ("Blinkit Express", "blinkit.retail", "P2M", 100, 1800),
    ("Reliance Smart Bazaar", "reliance.retail", "P2M", 400, 4500),
    ("Apollo Pharmacy", "apollo.meds", "P2M", 80, 2500),
    ("Local Kirana Store", "kirana.merchant", "QR_SCAN", 20, 800),
    ("Chai Point Express", "chaipoint.pos", "QR_SCAN", 15, 200),
    ("Uber India", "uber.rides", "P2M", 80, 1500),
    ("Indian Oil Petrol", "indianoil.fuel", "QR_SCAN", 200, 3500),
    ("Tata Power Electricity", "tatapower.bill", "BILL_PAY", 500, 6000),
    ("Airtel Postpaid", "airtel.bills", "BILL_PAY", 199, 1499),
    ("Amazon India", "amazon.pay", "P2M", 300, 15000),
    ("Flipkart Payments", "flipkart.pay", "P2M", 300, 18000),
    ("Croma Electronics", "croma.retail", "P2M", 1500, 45000),
    ("MakeMyTrip Flight", "makemytrip.travel", "P2M", 3000, 60000)
]

def generate_user_profile(user_idx):
    first = FIRST_NAMES[user_idx % len(FIRST_NAMES)]
    last = LAST_NAMES[(user_idx * 3) % len(LAST_NAMES)]
    name = f"{first} {last}"
    handle = random.choice(UPI_HANDLES)
    clean_id = f"{first.lower()}.{last.lower()}{random.randint(10, 999)}{handle}"
    home_city = random.choice(INDIAN_CITIES)
    account_age = random.randint(30, 2000) # days
    avg_amt = random.choice([250, 450, 750, 1200, 2500, 5000])
    std_amt = avg_amt * random.uniform(0.3, 0.7)
    typical_device = f"DEV_IND_{user_idx:05d}_{random.choice(['SM-S918B', 'iPhone15,2', 'OnePlus11', 'RedmiNote12', 'Pixel8'])}"
    return {
        "user_id": f"USR_{user_idx:05d}",
        "name": name,
        "upi_id": clean_id,
        "home_city": home_city,
        "account_age_days": account_age,
        "avg_amount": avg_amt,
        "std_amount": std_amt,
        "device_id": typical_device,
        "device_trust_score": round(random.uniform(0.85, 0.99), 2)
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_synthetic_upi_dataset(num_records=10000):
    print(f"Generating {num_records} synthetic UPI transaction records...")
    
    # 1. Create a pool of 800 regular users
    users = [generate_user_profile(i) for i in range(800)]
    
    # Track previous transaction state per user
    user_state = {}
    for u in users:
        city_coords = LOCATIONS[u["home_city"]]
        user_state[u["upi_id"]] = {
            "last_time": datetime(2026, 8, 1, 0, 0, 0) + timedelta(minutes=random.randint(0, 1440)),
            "last_city": u["home_city"],
            "last_lat": city_coords["lat"],
            "last_lon": city_coords["lon"],
            "last_device": u["device_id"],
            "txn_history_amounts": []
        }
    
    records = []
    base_time = datetime(2026, 8, 1, 8, 0, 0)
    current_sim_time = base_time
    
    # Pre-determine fraud indices (approx 5.5% fraud rate = ~550 fraudulent txns)
    fraud_indices = set(random.sample(range(num_records), int(num_records * 0.055)))
    
    fraud_types = [
        "IMPOSSIBLE_TRAVEL",
        "VELOCITY_DRAIN_BURST",
        "LATE_NIGHT_DORMANT_SPIKE",
        "PHISHING_QR_COLLECT_SCAM",
        "MULE_FANOUT_SUSPICIOUS_VPN",
        "DEVICE_MISMATCH_MAX_TRANSFER"
    ]
    
    for i in range(num_records):
        txn_id = f"TXN_UPI_{202608000000 + i}"
        
        # Advance simulation time realistically (avg 2-5 minutes per step across network)
        current_sim_time += timedelta(seconds=random.randint(10, 180))
        
        # Pick random sender user
        sender = random.choice(users)
        s_upi = sender["upi_id"]
        s_state = user_state[s_upi]
        
        is_fraud_case = i in fraud_indices
        fraud_type = "NONE"
        
        # Diurnal time check (hour of day)
        hour = current_sim_time.hour
        
        if not is_fraud_case:
            # ----------------- NORMAL TRANSACTION -----------------
            is_fraud = 0
            txn_category = random.choices(["P2M", "QR_SCAN", "P2P", "BILL_PAY"], weights=[0.45, 0.30, 0.18, 0.07])[0]
            
            if txn_category in ["P2M", "QR_SCAN", "BILL_PAY"]:
                merchant = random.choice(MERCHANTS)
                m_name, m_handle_prefix, _, min_amt, max_amt = merchant
                m_handle = random.choice(MERCHANT_HANDLES)
                receiver_name = m_name
                receiver_upi = f"{m_handle_prefix}{m_handle}"
                # Realistic gamma / log-normal distribution for amount
                raw_amt = random.uniform(min_amt, max_amt)
                amount = round(raw_amt, 2)
                is_new_payee = 0 if random.random() < 0.65 else 1
            else: # P2P
                receiver_user = random.choice(users)
                while receiver_user["upi_id"] == s_upi:
                    receiver_user = random.choice(users)
                receiver_name = receiver_user["name"]
                receiver_upi = receiver_user["upi_id"]
                # Normal P2P amount around sender's average
                amount = max(10.0, round(random.gauss(sender["avg_amount"], sender["std_amount"]), 2))
                amount = min(amount, 25000.0)
                is_new_payee = 1 if random.random() < 0.35 else 0
            
            # Normal location: mostly home city (95%) or domestic travel (5%)
            if random.random() < 0.95:
                city = sender["home_city"]
            else:
                city = random.choice(INDIAN_CITIES)
            
            city_geo = LOCATIONS[city]
            # Small jitter in lat/lon for city area
            lat = city_geo["lat"] + random.uniform(-0.05, 0.05)
            lon = city_geo["lon"] + random.uniform(-0.05, 0.05)
            
            device_id = sender["device_id"]
            device_trust = sender["device_trust_score"]
            network_type = random.choices(["4G", "5G", "WIFI"], weights=[0.45, 0.40, 0.15])[0]
            failed_pin_attempts = 0 if random.random() < 0.98 else 1
            account_age = sender["account_age_days"]
            
        else:
            # ----------------- FRAUDULENT TRANSACTION -----------------
            is_fraud = 1
            fraud_type = random.choice(fraud_types)
            
            if fraud_type == "IMPOSSIBLE_TRAVEL":
                dest_choice = random.choice(["London_Anomaly", "Lagos_Anomaly", "Moscow_Anomaly"])
                city = dest_choice.replace("_Anomaly", "")
                geo = LOCATIONS[dest_choice]
                lat = geo["lat"] + random.uniform(-0.02, 0.02)
                lon = geo["lon"] + random.uniform(-0.02, 0.02)
                amount = round(random.uniform(15000, 85000), 2)
                txn_category = "P2P"
                receiver_name = "Unknown Overseas Account"
                receiver_upi = f"suspect.node{random.randint(100, 999)}@ybl"
                device_id = f"DEV_UNKNOWN_{random.randint(1000, 9999)}"
                device_trust = round(random.uniform(0.1, 0.35), 2)
                network_type = "VPN_SUSPICIOUS"
                failed_pin_attempts = random.choice([0, 1, 2])
                is_new_payee = 1
                account_age = sender["account_age_days"]
                
            elif fraud_type == "VELOCITY_DRAIN_BURST":
                city = sender["home_city"]
                geo = LOCATIONS[city]
                lat = geo["lat"] + random.uniform(-0.02, 0.02)
                lon = geo["lon"] + random.uniform(-0.02, 0.02)
                amount = round(random.uniform(40000, 99000), 2)
                txn_category = "P2P"
                receiver_name = f"Mule Conduit {random.randint(1, 50)}"
                receiver_upi = f"fastdrain.mule{random.randint(100, 999)}@paytm"
                device_id = sender["device_id"] if random.random() < 0.5 else f"DEV_CLONED_{random.randint(1000, 9999)}"
                device_trust = round(random.uniform(0.2, 0.5), 2)
                network_type = random.choice(["4G", "VPN_SUSPICIOUS"])
                failed_pin_attempts = random.choice([1, 2, 3])
                is_new_payee = 1
                account_age = sender["account_age_days"]
                
            elif fraud_type == "LATE_NIGHT_DORMANT_SPIKE":
                city = sender["home_city"]
                geo = LOCATIONS[city]
                lat = geo["lat"] + random.uniform(-0.01, 0.01)
                lon = geo["lon"] + random.uniform(-0.01, 0.01)
                amount = round(random.uniform(50000, 98000), 2)
                txn_category = "P2P"
                receiver_name = "Midnight Shadow Account"
                receiver_upi = f"nighttransact{random.randint(100, 999)}@apl"
                device_id = sender["device_id"]
                device_trust = round(random.uniform(0.4, 0.7), 2)
                network_type = "4G"
                failed_pin_attempts = random.choice([0, 1])
                is_new_payee = 1
                account_age = sender["account_age_days"]
                
            elif fraud_type == "PHISHING_QR_COLLECT_SCAM":
                city = random.choice(INDIAN_CITIES)
                geo = LOCATIONS[city]
                lat = geo["lat"] + random.uniform(-0.01, 0.01)
                lon = geo["lon"] + random.uniform(-0.01, 0.01)
                amount = round(random.uniform(9999, 49999), 2)
                txn_category = "COLLECT_REQUEST"
                receiver_name = "Cashback Rewards NPCI Official (Fake)"
                receiver_upi = f"claim.rewards.gov{random.randint(10, 99)}@axl"
                device_id = f"DEV_PHISH_{random.randint(1000, 9999)}"
                device_trust = round(random.uniform(0.15, 0.45), 2)
                network_type = "VPN_SUSPICIOUS"
                failed_pin_attempts = 0
                is_new_payee = 1
                account_age = sender["account_age_days"]
                
            elif fraud_type == "MULE_FANOUT_SUSPICIOUS_VPN":
                city = random.choice(INDIAN_CITIES)
                geo = LOCATIONS[city]
                lat = geo["lat"] + random.uniform(-0.01, 0.01)
                lon = geo["lon"] + random.uniform(-0.01, 0.01)
                amount = round(random.uniform(30000, 75000), 2)
                txn_category = "P2P"
                receiver_name = "Layering Mule Hub"
                receiver_upi = f"mulelayer{random.randint(10, 99)}@ibl"
                device_id = f"DEV_EMULATOR_{random.randint(1000, 9999)}"
                device_trust = round(random.uniform(0.05, 0.25), 2)
                network_type = "VPN_SUSPICIOUS"
                failed_pin_attempts = 2
                is_new_payee = 1
                account_age = random.randint(1, 14)
                
            else: # DEVICE_MISMATCH_MAX_TRANSFER
                city = sender["home_city"]
                geo = LOCATIONS[city]
                lat = geo["lat"] + random.uniform(-0.01, 0.01)
                lon = geo["lon"] + random.uniform(-0.01, 0.01)
                amount = 100000.00
                txn_category = "P2P"
                receiver_name = "Immediate Cashout Broker"
                receiver_upi = f"cashout.crypto{random.randint(10, 99)}@oksbi"
                device_id = f"DEV_UNKNOWN_{random.randint(8000, 9999)}"
                device_trust = 0.10
                network_type = "VPN_SUSPICIOUS"
                failed_pin_attempts = 3
                is_new_payee = 1
                account_age = sender["account_age_days"]
        
        # Calculate time delta & spatial distance from user's last transaction
        time_delta_sec = max(1.0, (current_sim_time - s_state["last_time"]).total_seconds())
        time_delta_hours = time_delta_sec / 3600.0
        dist_km = haversine_distance(s_state["last_lat"], s_state["last_lon"], lat, lon)
        speed_kmh = dist_km / time_delta_hours if time_delta_hours > 0 else 0.0
        
        # Record state update
        s_state["last_time"] = current_sim_time
        s_state["last_lat"] = lat
        s_state["last_lon"] = lon
        s_state["last_city"] = city
        s_state["last_device"] = device_id
        s_state["txn_history_amounts"].append(amount)
        
        # Historical user stats
        user_history_count = len(s_state["txn_history_amounts"])
        user_avg_amt = sender["avg_amount"]
        amt_to_avg_ratio = round(amount / user_avg_amt, 3) if user_avg_amt > 0 else 1.0
        
        record = {
            "transaction_id": txn_id,
            "timestamp": current_sim_time.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "sender_upi_id": s_upi,
            "sender_name": sender["name"],
            "receiver_upi_id": receiver_upi,
            "receiver_name": receiver_name,
            "amount": amount,
            "transaction_type": txn_category,
            "device_id": device_id,
            "location_city": city,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "network_type": network_type,
            "device_trust_score": device_trust,
            "failed_pin_attempts": failed_pin_attempts,
            "sender_account_age_days": account_age,
            "is_new_payee": is_new_payee,
            "time_since_last_txn_sec": round(time_delta_sec, 1),
            "distance_from_last_txn_km": round(dist_km, 2),
            "travel_speed_kmh": round(speed_kmh, 2),
            "amount_to_user_avg_ratio": amt_to_avg_ratio,
            "user_historical_txns": user_history_count,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    
    # Save to data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "synthetic_upi_10k.csv")
    json_path = os.path.join(data_dir, "synthetic_upi_10k.json")
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", date_format="iso", indent=2)
    
    print(f"Generated {len(df)} transactions.")
    print(f"Saved to:\n - {csv_path}\n - {json_path}")
    print(f"Total Fraud Count: {df['is_fraud'].sum()} ({df['is_fraud'].mean() * 100:.2f}%)")
    
    return df

if __name__ == "__main__":
    generate_synthetic_upi_dataset(10000)
