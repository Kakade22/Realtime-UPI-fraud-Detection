"""
Technique 1: Simple IQR (Interquartile Range) Anomaly Detection
Uses nonparametric statistical bounds (Q1, Q3, IQR = Q3 - Q1) to identify
distributional outliers in transaction amount, user relative ratio, and travel velocity.
"""

import numpy as np
import pandas as pd

class IQRAnomalyDetector:
    def __init__(self, multiplier=1.5, extreme_multiplier=3.0):
        self.multiplier = multiplier
        self.extreme_multiplier = extreme_multiplier
        self.is_fitted = False
        self.bounds = {}
        self.user_stats = {}

    def fit(self, df: pd.DataFrame):
        """Fits IQR boundaries on the baseline transaction dataset."""
        normal_df = df[df["is_fraud"] == 0] if "is_fraud" in df.columns else df
        
        # 1. Global amount bounds
        amt = normal_df["amount"].values
        q1_amt, q3_amt = np.percentile(amt, [25, 75])
        iqr_amt = q3_amt - q1_amt
        self.bounds["amount"] = {
            "q1": float(q1_amt),
            "q3": float(q3_amt),
            "iqr": float(iqr_amt),
            "upper_mild": float(q3_amt + self.multiplier * iqr_amt),
            "upper_extreme": float(q3_amt + self.extreme_multiplier * iqr_amt)
        }

        # 2. Ratio to user average bounds
        ratios = normal_df["amount_to_user_avg_ratio"].values
        q1_r, q3_r = np.percentile(ratios, [25, 75])
        iqr_r = q3_r - q1_r
        self.bounds["ratio"] = {
            "q1": float(q1_r),
            "q3": float(q3_r),
            "iqr": float(iqr_r),
            "upper_mild": float(q3_r + self.multiplier * iqr_r),
            "upper_extreme": float(q3_r + self.extreme_multiplier * iqr_r)
        }

        # 3. Travel speed bounds (for non-zero travel)
        speeds = normal_df["travel_speed_kmh"].values
        speeds_nonzero = speeds[speeds > 0]
        if len(speeds_nonzero) > 0:
            q1_s, q3_s = np.percentile(speeds_nonzero, [25, 75])
            iqr_s = q3_s - q1_s
        else:
            q1_s, q3_s, iqr_s = 0, 100, 100
        self.bounds["speed"] = {
            "q1": float(q1_s),
            "q3": float(q3_s),
            "iqr": float(iqr_s),
            "upper_mild": float(max(200.0, q3_s + self.multiplier * iqr_s)),
            "upper_extreme": float(max(500.0, q3_s + self.extreme_multiplier * iqr_s))
        }

        # 4. Per-user statistical profiles
        for upi_id, grp in normal_df.groupby("sender_upi_id"):
            u_amts = grp["amount"].values
            if len(u_amts) >= 4:
                u_q1, u_q3 = np.percentile(u_amts, [25, 75])
                u_iqr = u_q3 - u_q1
                self.user_stats[upi_id] = {
                    "q1": float(u_q1),
                    "q3": float(u_q3),
                    "iqr": float(u_iqr),
                    "upper_mild": float(u_q3 + self.multiplier * u_iqr),
                    "upper_extreme": float(u_q3 + self.extreme_multiplier * u_iqr),
                    "mean": float(np.mean(u_amts))
                }

        self.is_fitted = True
        return self

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Fast vectorized scoring for batch evaluation."""
        amounts = df["amount"].astype(float).values
        ratios = df["amount_to_user_avg_ratio"].astype(float).values
        speeds = df.get("travel_speed_kmh", pd.Series(np.zeros(len(df)))).astype(float).values

        amt_upper_ext = self.bounds["amount"]["upper_extreme"]
        amt_upper_mild = self.bounds["amount"]["upper_mild"]
        rat_upper_ext = self.bounds["ratio"]["upper_extreme"]
        rat_upper_mild = self.bounds["ratio"]["upper_mild"]
        spd_upper_ext = self.bounds["speed"]["upper_extreme"]
        spd_upper_mild = self.bounds["speed"]["upper_mild"]

        amt_scores = np.where(amounts > amt_upper_ext, 1.0, np.where(amounts > amt_upper_mild, 0.65, 0.0))
        rat_scores = np.where(ratios > rat_upper_ext, 0.9, np.where(ratios > rat_upper_mild, 0.5, 0.0))
        spd_scores = np.where(speeds > spd_upper_ext, 1.0, np.where(speeds > spd_upper_mild, 0.6, 0.0))

        iqr_scores = np.maximum(amt_scores, np.maximum(rat_scores, spd_scores))
        return iqr_scores

    def predict_single(self, txn: dict) -> dict:
        """Evaluates a single transaction against IQR boundaries."""
        if not self.is_fitted:
            raise RuntimeError("IQRAnomalyDetector must be fitted before predict_single.")

        amount = float(txn.get("amount", 0.0))
        ratio = float(txn.get("amount_to_user_avg_ratio", 1.0))
        speed = float(txn.get("travel_speed_kmh", 0.0))
        upi_id = txn.get("sender_upi_id", "")

        anomalies = []
        score_components = []

        # Check Global Amount IQR
        amt_bounds = self.bounds["amount"]
        if amount > amt_bounds["upper_extreme"]:
            score_components.append(1.0)
            anomalies.append(f"Amount ₹{amount:,.2f} is an extreme global IQR outlier (> ₹{amt_bounds['upper_extreme']:,.2f})")
        elif amount > amt_bounds["upper_mild"]:
            score_components.append(0.65)
            anomalies.append(f"Amount ₹{amount:,.2f} exceeds global IQR upper bound (₹{amt_bounds['upper_mild']:,.2f})")
        else:
            score_components.append(0.0)

        # Check User-Specific Amount IQR if available
        if upi_id in self.user_stats:
            u_stat = self.user_stats[upi_id]
            if u_stat["iqr"] > 0:
                if amount > u_stat["upper_extreme"]:
                    score_components.append(1.0)
                    anomalies.append(f"Amount ₹{amount:,.2f} is 3x IQR outlier for user's typical spending (threshold ₹{u_stat['upper_extreme']:,.2f})")
                elif amount > u_stat["upper_mild"]:
                    score_components.append(0.7)
                    anomalies.append(f"Amount ₹{amount:,.2f} is 1.5x IQR outlier for user's profile (threshold ₹{u_stat['upper_mild']:,.2f})")
                else:
                    score_components.append(0.0)

        # Check Amount to User Avg Ratio IQR
        rat_bounds = self.bounds["ratio"]
        if ratio > rat_bounds["upper_extreme"]:
            score_components.append(0.9)
            anomalies.append(f"Amount is {ratio:.1f}x of user average (extreme ratio anomaly)")
        elif ratio > rat_bounds["upper_mild"]:
            score_components.append(0.5)
            anomalies.append(f"Amount is {ratio:.1f}x of user average (ratio outlier)")
        else:
            score_components.append(0.0)

        # Check Travel Speed IQR
        spd_bounds = self.bounds["speed"]
        if speed > spd_bounds["upper_extreme"]:
            score_components.append(1.0)
            anomalies.append(f"Travel speed {speed:.1f} km/h is an extreme speed anomaly (> {spd_bounds['upper_extreme']:.0f} km/h)")
        elif speed > spd_bounds["upper_mild"]:
            score_components.append(0.6)
            anomalies.append(f"Travel speed {speed:.1f} km/h exceeds normal mobility threshold ({spd_bounds['upper_mild']:.0f} km/h)")
        else:
            score_components.append(0.0)

        # Aggregated IQR Anomaly Score [0, 1]
        iqr_score = float(max(score_components)) if score_components else 0.0
        is_anomaly = iqr_score >= 0.60

        return {
            "technique": "Simple IQR Technique",
            "anomaly_score": round(iqr_score, 4),
            "is_anomaly": bool(is_anomaly),
            "reasons": anomalies,
            "bounds_summary": {
                "amount_mild_threshold": amt_bounds["upper_mild"],
                "amount_extreme_threshold": amt_bounds["upper_extreme"],
                "speed_extreme_threshold": spd_bounds["upper_extreme"]
            }
        }
