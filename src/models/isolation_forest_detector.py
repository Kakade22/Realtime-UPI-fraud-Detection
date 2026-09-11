"""
Technique 2: Multi-Dimensional Isolation Forest Anomaly Detection
Uses Scikit-Learn IsolationForest on scaled multi-dimensional feature vectors:
financial magnitude, cyclical temporal patterns, geospatial velocity,
device security score, authentication failures, and network integrity.
"""

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationForestDetector:
    def __init__(self, contamination=0.055, n_estimators=150, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples="auto",
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = [
            "log_amount",
            "amount_to_avg_ratio",
            "hour_sin",
            "hour_cos",
            "log_travel_speed",
            "log_distance",
            "log_time_delta",
            "device_trust_score",
            "failed_pin_attempts",
            "is_new_payee",
            "is_vpn",
            "is_collect_request",
            "log_account_age"
        ]

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms raw transaction columns into standardized feature vectors."""
        amounts = df["amount"].astype(float).values
        ratios = df["amount_to_user_avg_ratio"].astype(float).values
        
        # Hour diurnal cyclical encoding
        if "hour" in df.columns:
            hours = df["hour"].astype(float).values
        else:
            hours = pd.to_datetime(df["timestamp"]).dt.hour.values
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        
        speeds = df.get("travel_speed_kmh", pd.Series(np.zeros(len(df)))).astype(float).values
        dists = df.get("distance_from_last_txn_km", pd.Series(np.zeros(len(df)))).astype(float).values
        time_deltas = df.get("time_since_last_txn_sec", pd.Series(np.ones(len(df)) * 3600)).astype(float).values
        
        trust_scores = df.get("device_trust_score", pd.Series(np.ones(len(df)) * 0.9)).astype(float).values
        pin_fails = df.get("failed_pin_attempts", pd.Series(np.zeros(len(df)))).astype(float).values
        new_payees = df.get("is_new_payee", pd.Series(np.zeros(len(df)))).astype(float).values
        
        is_vpn = (df.get("network_type", pd.Series(["4G"] * len(df))) == "VPN_SUSPICIOUS").astype(float).values
        is_collect = (df.get("transaction_type", pd.Series(["P2P"] * len(df))) == "COLLECT_REQUEST").astype(float).values
        
        account_ages = df.get("sender_account_age_days", pd.Series(np.ones(len(df)) * 180)).astype(float).values

        # Log transformations for heavily skewed positive variables
        log_amount = np.log1p(np.maximum(0, amounts))
        log_travel_speed = np.log1p(np.maximum(0, speeds))
        log_distance = np.log1p(np.maximum(0, dists))
        log_time_delta = np.log1p(np.maximum(0, time_deltas))
        log_account_age = np.log1p(np.maximum(0, account_ages))

        feat_matrix = np.column_stack([
            log_amount,
            ratios,
            hour_sin,
            hour_cos,
            log_travel_speed,
            log_distance,
            log_time_delta,
            trust_scores,
            pin_fails,
            new_payees,
            is_vpn,
            is_collect,
            log_account_age
        ])
        return feat_matrix

    def fit(self, df: pd.DataFrame):
        """Fits the Isolation Forest on transaction features."""
        X = self._extract_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def _extract_single_features(self, txn: dict) -> np.ndarray:
        """Fast direct feature array builder without DataFrame overhead (< 0.05ms)."""
        amount = float(txn.get("amount", 0.0))
        ratio = float(txn.get("amount_to_user_avg_ratio", 1.0))
        hour = float(txn.get("hour", 12))
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        speed = float(txn.get("travel_speed_kmh", 0.0))
        dist = float(txn.get("distance_from_last_txn_km", 0.0))
        time_delta = float(txn.get("time_since_last_txn_sec", 3600.0))
        trust = float(txn.get("device_trust_score", 0.9))
        pin_fail = float(txn.get("failed_pin_attempts", 0))
        new_payee = float(txn.get("is_new_payee", 0))
        is_vpn = 1.0 if txn.get("network_type") == "VPN_SUSPICIOUS" else 0.0
        is_collect = 1.0 if txn.get("transaction_type") == "COLLECT_REQUEST" else 0.0
        acc_age = float(txn.get("sender_account_age_days", 180))

        feat = np.array([[
            math.log1p(max(0, amount)),
            ratio,
            hour_sin,
            hour_cos,
            math.log1p(max(0, speed)),
            math.log1p(max(0, dist)),
            math.log1p(max(0, time_delta)),
            trust,
            pin_fail,
            new_payee,
            is_vpn,
            is_collect,
            math.log1p(max(0, acc_age))
        ]], dtype=np.float64)
        return feat

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Fast vectorized batch scoring for dataset evaluation (< 0.05s for 10k rows)."""
        X = self._extract_features(df)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        risk_scores = 1.0 / (1.0 + np.exp(raw_scores * 12.0))
        return np.clip(risk_scores, 0.0, 1.0)

    def predict_single(self, txn: dict) -> dict:
        """Scores a single incoming transaction and provides interpretability."""
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before predict_single.")

        X = self._extract_single_features(txn)
        X_scaled = self.scaler.transform(X)

        # Scikit-learn raw decision score (negative = anomaly, positive = normal)
        raw_score = float(self.model.decision_function(X_scaled)[0])
        pred_label = int(self.model.predict(X_scaled)[0])  # -1 for anomaly, 1 for normal

        # Calibrate raw score into smooth risk score [0, 1]
        # Raw score ~ +0.15 is very normal, -0.25 is extreme anomaly
        # Using calibrated sigmoid mapping
        risk_score = 1.0 / (1.0 + math.exp(raw_score * 12.0))
        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        is_anomaly = bool(pred_label == -1 or risk_score >= 0.55)

        # Explainability: identify top deviant feature dimensions compared to standardized distribution
        deviations = np.abs(X_scaled[0])
        top_deviant_indices = np.argsort(deviations)[::-1][:3]
        
        reasons = []
        for idx in top_deviant_indices:
            feat_name = self.feature_names[idx]
            val = X[0, idx]
            dev = deviations[idx]
            if dev > 1.8:  # Significant feature deviation
                if "amount" in feat_name:
                    reasons.append(f"Unusual transaction value deviation (Z={dev:.1f})")
                elif "speed" in feat_name:
                    reasons.append(f"Abnormal geolocation velocity deviation (Z={dev:.1f})")
                elif "trust" in feat_name:
                    reasons.append(f"Compromised device trust signature (score={txn.get('device_trust_score', 0):.2f})")
                elif "vpn" in feat_name and val > 0:
                    reasons.append("High-risk VPN network tunnel detected")
                elif "collect" in feat_name and val > 0:
                    reasons.append("High-risk reverse collect request pattern")
                elif "pin" in feat_name and val > 0:
                    reasons.append(f"Repeated PIN authentication failures ({int(txn.get('failed_pin_attempts', 0))})")
                elif "time_delta" in feat_name:
                    reasons.append("Abnormal inter-transaction timing interval")

        if not reasons and is_anomaly:
            reasons.append("Multi-feature multidimensional isolation anomaly")

        return {
            "technique": "Isolation Forest",
            "anomaly_score": round(risk_score, 4),
            "raw_decision_score": round(raw_score, 4),
            "is_anomaly": is_anomaly,
            "reasons": reasons
        }
