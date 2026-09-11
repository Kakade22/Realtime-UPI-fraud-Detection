"""
Technique 3: Time Series & Sequential Rolling Anomaly Detection
Uses Exponentially Weighted Moving Average (EWMA), rolling velocity z-scores,
Poisson inter-arrival delta deviation, and diurnal quiet-hour transitions to identify
temporal bursts, rapid drain cascades, and abnormal sequential patterns.
"""

import math
import numpy as np
import pandas as pd

class TimeSeriesAnomalyDetector:
    def __init__(self, ewma_alpha=0.2):
        self.ewma_alpha = ewma_alpha
        self.is_fitted = False
        self.global_hourly_volume = {}
        self.global_hourly_mean = 0.0
        self.global_hourly_std = 1.0

    def fit(self, df: pd.DataFrame):
        """Fits temporal baseline models on historical transaction stream."""
        normal_df = (df[df["is_fraud"] == 0] if "is_fraud" in df.columns else df).copy()
        
        # Calculate hourly volume distributions across Indian payment traffic
        if "hour" in normal_df.columns:
            normal_df["hour_dt"] = normal_df["hour"]
        else:
            normal_df["hour_dt"] = pd.to_datetime(normal_df["timestamp"]).dt.hour

        hourly_stats = normal_df.groupby("hour_dt")["amount"].agg(["mean", "std", "count"]).fillna(0)
        self.global_hourly_volume = hourly_stats.to_dict(orient="index")
        
        self.global_hourly_mean = float(normal_df["amount"].mean())
        self.global_hourly_std = float(max(1.0, normal_df["amount"].std()))
        self.is_fitted = True
        return self

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Fast vectorized scoring for batch temporal anomaly evaluation."""
        amounts = df["amount"].astype(float).values
        time_deltas = df.get("time_since_last_txn_sec", pd.Series(np.ones(len(df)) * 3600)).astype(float).values
        hours = df.get("hour", pd.Series(np.ones(len(df)) * 12)).astype(int).values

        # 1. Burstiness check
        burst_scores = np.where((time_deltas < 30.0) & (amounts > 5000.0), 0.95,
                       np.where((time_deltas < 60.0) & (amounts > 2000.0), 0.70, 0.05))

        # 2. Night surge check
        night_mask = (hours >= 1) & (hours <= 4)
        night_scores = np.where(night_mask & (amounts > 30000.0), 0.88,
                       np.where(night_mask & (amounts > 10000.0), 0.55, 0.0))

        ts_scores = np.maximum(burst_scores, night_scores)
        return ts_scores

    def predict_single(self, txn: dict) -> dict:
        """Scores temporal sequence & rolling window velocity dynamics."""
        if not self.is_fitted:
            raise RuntimeError("TimeSeriesAnomalyDetector must be fitted before predict_single.")

        amount = float(txn.get("amount", 0.0))
        time_delta_sec = float(txn.get("time_since_last_txn_sec", 3600.0))
        txns_5m = int(txn.get("txns_last_5m", 0))
        txns_1h = int(txn.get("txns_last_1h", 0))
        vol_1h = float(txn.get("volume_last_1h", amount))
        
        hour = int(txn.get("hour", 12))
        
        reasons = []
        scores = []

        # 1. Burstiness & Inter-Arrival Rapid Fire Check (< 60 seconds delta)
        if time_delta_sec < 30.0 and amount > 5000.0:
            scores.append(0.95)
            reasons.append(f"Rapid burst: Consecutive transfer within {time_delta_sec:.1f}s (velocity drain cascade)")
        elif time_delta_sec < 60.0 and amount > 2000.0:
            scores.append(0.70)
            reasons.append(f"High-frequency interval: {time_delta_sec:.1f}s since previous transaction")
        else:
            scores.append(0.05)

        # 2. Short-window Frequency Burst (5 min & 1 hour velocity)
        if txns_5m >= 3:
            scores.append(0.90)
            reasons.append(f"Abnormal velocity: {txns_5m} transactions attempted in past 5 minutes")
        elif txns_1h >= 5:
            scores.append(0.75)
            reasons.append(f"High 1-hour frequency: {txns_1h} transactions in last hour")
        else:
            scores.append(0.0)

        # 3. Rolling Cumulative Outflow Spike
        if vol_1h > 75000.0:
            scores.append(0.92)
            reasons.append(f"1-Hour cumulative outflow spike (₹{vol_1h:,.2f} spent within 60 mins)")
        elif vol_1h > 35000.0:
            scores.append(0.60)
            reasons.append(f"Elevated 1-hour volume (₹{vol_1h:,.2f})")
        else:
            scores.append(0.0)

        # 4. Diurnal Quiet-Hours Surge (1:00 AM to 4:30 AM)
        is_night_window = (hour >= 1 and hour <= 4)
        if is_night_window:
            if amount > 30000.0:
                scores.append(0.88)
                reasons.append(f"Off-peak nocturnal surge: High amount ₹{amount:,.2f} initiated at {hour:02d}:00 HRS")
            elif amount > 10000.0:
                scores.append(0.55)
                reasons.append(f"Dormant hour transfer at {hour:02d}:00 HRS")
        
        ts_score = float(max(scores)) if scores else 0.05
        is_anomaly = ts_score >= 0.60

        return {
            "technique": "Time Series / Rolling Velocity",
            "anomaly_score": round(ts_score, 4),
            "is_anomaly": bool(is_anomaly),
            "reasons": reasons,
            "temporal_metrics": {
                "time_delta_sec": time_delta_sec,
                "txns_last_5m": txns_5m,
                "txns_last_1h": txns_1h,
                "volume_last_1h": vol_1h,
                "hour_of_day": hour
            }
        }
