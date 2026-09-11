"""
Real-Time Stateful User Tracking Engine
Tracks in-memory sliding window history per user to compute dynamic features:
- Velocity (transactions in last 5m, 1h)
- Cumulative volume (INR in last 1h)
- Geolocation jump speed (km/h via Haversine)
- Historical user profiling (moving average, variance, baseline IQR)
"""

import math
from datetime import datetime, timedelta
from collections import defaultdict, deque

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class UserProfile:
    def __init__(self, sender_upi_id, initial_city=None, initial_lat=None, initial_lon=None, initial_device=None):
        self.upi_id = sender_upi_id
        self.last_timestamp = None
        self.last_city = initial_city or "Mumbai"
        self.last_lat = initial_lat if initial_lat is not None else 19.0760
        self.last_lon = initial_lon if initial_lon is not None else 72.8777
        self.known_devices = {initial_device} if initial_device else set()
        
        # Sliding window of recent transactions (within last 24h)
        self.recent_txns = deque()
        # Historical amounts for statistical distribution
        self.all_amounts = []
        self.account_age_days = 180

    def update(self, txn_dict):
        """Updates user state with a new transaction and returns engineered real-time features."""
        ts_str = txn_dict.get("timestamp")
        if isinstance(ts_str, str):
            try:
                current_time = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                current_time = datetime.now()
        elif isinstance(ts_str, datetime):
            current_time = ts_str
        else:
            current_time = datetime.now()

        amount = float(txn_dict.get("amount", 0.0))
        lat = float(txn_dict.get("latitude", self.last_lat))
        lon = float(txn_dict.get("longitude", self.last_lon))
        city = txn_dict.get("location_city", self.last_city)
        device_id = txn_dict.get("device_id", "")
        if "sender_account_age_days" in txn_dict:
            self.account_age_days = int(txn_dict["sender_account_age_days"])

        # 1. Temporal delta & Spatial speed
        if self.last_timestamp is not None:
            time_delta_sec = max(1.0, (current_time - self.last_timestamp).total_seconds())
        else:
            time_delta_sec = 3600.0  # Default 1 hour if first time seen

        time_delta_hours = time_delta_sec / 3600.0
        dist_km = haversine_distance(self.last_lat, self.last_lon, lat, lon)
        speed_kmh = dist_km / time_delta_hours if time_delta_hours > 0 else 0.0

        # 2. Prune transactions older than 24 hours from rolling deque
        cutoff_24h = current_time - timedelta(hours=24)
        while self.recent_txns and self.recent_txns[0]["time"] < cutoff_24h:
            self.recent_txns.popleft()

        # 3. Calculate sliding window metrics
        cutoff_5m = current_time - timedelta(minutes=5)
        cutoff_1h = current_time - timedelta(hours=1)

        txns_last_5m = sum(1 for t in self.recent_txns if t["time"] >= cutoff_5m)
        txns_last_1h = sum(1 for t in self.recent_txns if t["time"] >= cutoff_1h)
        volume_last_1h = sum(t["amount"] for t in self.recent_txns if t["time"] >= cutoff_1h)

        # 4. Historical baseline stats
        if self.all_amounts:
            user_avg = sum(self.all_amounts) / len(self.all_amounts)
            amt_to_avg_ratio = round(amount / user_avg, 3) if user_avg > 0 else 1.0
        else:
            user_avg = amount
            amt_to_avg_ratio = 1.0

        is_new_device = 1 if (self.known_devices and device_id not in self.known_devices) else 0

        # 5. Append to history & update state
        self.recent_txns.append({
            "time": current_time,
            "amount": amount,
            "lat": lat,
            "lon": lon
        })
        self.all_amounts.append(amount)
        if device_id:
            self.known_devices.add(device_id)

        self.last_timestamp = current_time
        self.last_lat = lat
        self.last_lon = lon
        self.last_city = city

        return {
            "time_since_last_txn_sec": round(time_delta_sec, 1),
            "distance_from_last_txn_km": round(dist_km, 2),
            "travel_speed_kmh": round(speed_kmh, 2),
            "amount_to_user_avg_ratio": amt_to_avg_ratio,
            "txns_last_5m": txns_last_5m,
            "txns_last_1h": txns_last_1h,
            "volume_last_1h": round(volume_last_1h, 2),
            "user_historical_txns": len(self.all_amounts),
            "is_new_device": is_new_device
        }

class StateEngine:
    def __init__(self):
        self.users = {}

    def get_user_profile(self, upi_id, **kwargs):
        if upi_id not in self.users:
            self.users[upi_id] = UserProfile(upi_id, **kwargs)
        return self.users[upi_id]

    def process_transaction(self, txn_dict):
        upi_id = txn_dict.get("sender_upi_id", "unknown@upi")
        profile = self.get_user_profile(
            upi_id,
            initial_city=txn_dict.get("location_city"),
            initial_lat=txn_dict.get("latitude"),
            initial_lon=txn_dict.get("longitude"),
            initial_device=txn_dict.get("device_id")
        )
        engineered_feats = profile.update(txn_dict)
        # Merge computed real-time features into transaction payload
        merged_txn = dict(txn_dict)
        merged_txn.update(engineered_feats)
        return merged_txn
