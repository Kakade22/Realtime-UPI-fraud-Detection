"""
Hybrid Ensemble & Explainability Engine
Combines:
1. Simple IQR Statistical Outlier Detection
2. Scikit-Learn Isolation Forest Multi-Dimensional Scoring
3. Time Series & Rolling Velocity / EWMA Anomaly Detection
4. Hard Deterministic Security Guardrails (Impossible Travel, Phishing Keywords, Brute-force PIN)

Produces calibrated 0-100 Fraud Risk Scores, actionable decision tiers,
and human-understandable explanation tags for instant user warning prompts.
"""

import time
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

from .iqr_detector import IQRAnomalyDetector
from .isolation_forest_detector import IsolationForestDetector
from .timeseries_detector import TimeSeriesAnomalyDetector

class UPIFraudEnsembleEngine:
    def __init__(self):
        self.iqr_model = IQRAnomalyDetector()
        self.iso_model = IsolationForestDetector()
        self.ts_model = TimeSeriesAnomalyDetector()
        self.is_fitted = False
        self.evaluation_metrics = {}

    def fit(self, df: pd.DataFrame):
        """Fits all three anomaly detection models on the synthetic dataset."""
        print("Training Simple IQR Detector...")
        self.iqr_model.fit(df)

        print("Training Isolation Forest Detector...")
        self.iso_model.fit(df)

        print("Training Time Series / Rolling Velocity Detector...")
        self.ts_model.fit(df)

        self.is_fitted = True
        print("Evaluating baseline metrics across 10,000 transactions...")
        self.evaluation_metrics = self.evaluate_dataset(df)
        print("Model training & benchmarking complete.")
        return self

    def predict_single(self, txn: dict) -> dict:
        """Processes a single incoming UPI transaction in real-time (< 5ms)."""
        if not self.is_fitted:
            raise RuntimeError("UPIFraudEnsembleEngine must be fitted before predict_single.")

        t0 = time.perf_counter()

        # Run individual detectors
        iqr_res = self.iqr_model.predict_single(txn)
        iso_res = self.iso_model.predict_single(txn)
        ts_res = self.ts_model.predict_single(txn)

        iqr_score = iqr_res["anomaly_score"]
        iso_score = iso_res["anomaly_score"]
        ts_score = ts_res["anomaly_score"]

        # Deterministic Security Guardrails
        speed_kmh = float(txn.get("travel_speed_kmh", 0.0))
        dist_km = float(txn.get("distance_from_last_txn_km", 0.0))
        failed_pins = int(txn.get("failed_pin_attempts", 0))
        device_trust = float(txn.get("device_trust_score", 0.9))
        network = txn.get("network_type", "4G")
        txn_type = txn.get("transaction_type", "P2P")
        amount = float(txn.get("amount", 0.0))

        rule_overrides = []
        rule_boost = 0.0

        # Rule 1: Impossible Physical Velocity (> 800 km/h)
        if speed_kmh > 800.0 and dist_km > 300.0:
            rule_boost = max(rule_boost, 0.95)
            rule_overrides.append(f"Impossible Travel: {speed_kmh:,.0f} km/h jump ({dist_km:,.0f} km away)")

        # Rule 2: Multiple PIN failures + VPN + High Amount
        if failed_pins >= 2 and network == "VPN_SUSPICIOUS" and amount > 25000:
            rule_boost = max(rule_boost, 0.92)
            rule_overrides.append(f"Credential Brute-Force & VPN detected ({failed_pins} failed PIN attempts)")

        # Rule 3: Collect Request Phishing Pattern
        if txn_type == "COLLECT_REQUEST" and (device_trust < 0.5 or network == "VPN_SUSPICIOUS"):
            rule_boost = max(rule_boost, 0.88)
            rule_overrides.append("Suspected UPI Collect Request Phishing / Fake Cashback Scam")

        # Rule 4: Zero Trust Device Max Limit Transfer
        if device_trust < 0.2 and amount >= 90000:
            rule_boost = max(rule_boost, 0.95)
            rule_overrides.append(f"Unverified Device attempting near-limit transfer (₹{amount:,.2f})")

        # Weighted Ensemble Calculation
        # Weights: Isolation Forest (0.45), Time Series (0.30), IQR (0.25)
        raw_weighted_score = (0.45 * iso_score) + (0.30 * ts_score) + (0.25 * iqr_score)
        
        # Apply rule boost if triggered
        final_probability = max(raw_weighted_score, rule_boost)
        final_probability = min(1.0, max(0.0, final_probability))

        risk_score_100 = int(round(final_probability * 100))

        # Risk Classification & Instant Action Determination
        if risk_score_100 >= 70:
            risk_level = "FRAUD"
            action = "BLOCK_INSTANTLY"
            action_title = "🛑 Transaction Blocked - Fraud Warning"
            action_message = "Suspicious activity detected. Transaction halted immediately to protect your bank account."
        elif risk_score_100 >= 35:
            risk_level = "CAUTION"
            action = "WARN_USER_STEP_UP"
            action_title = "⚠️ Security Warning - Verify Payee"
            action_message = "Unusual transaction parameters. Please verify the recipient identity and purpose before authorizing."
        else:
            risk_level = "SAFE"
            action = "ALLOW"
            action_title = "✅ Payment Approved"
            action_message = "Transaction passed all security heuristics and anomaly checks."

        # Compile comprehensive explainability tags
        all_reasons = []
        all_reasons.extend(rule_overrides)
        all_reasons.extend(iso_res["reasons"])
        all_reasons.extend(ts_res["reasons"])
        all_reasons.extend(iqr_res["reasons"])
        # Deduplicate while preserving order
        seen = set()
        dedup_reasons = []
        for r in all_reasons:
            if r not in seen:
                seen.add(r)
                dedup_reasons.append(r)

        if not dedup_reasons:
            if risk_level == "SAFE":
                dedup_reasons.append("Behavior matches normal spending profile & registered device")
            else:
                dedup_reasons.append("Multi-factor composite risk deviation")

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "transaction_id": txn.get("transaction_id", "UNKNOWN"),
            "sender_upi_id": txn.get("sender_upi_id", ""),
            "receiver_upi_id": txn.get("receiver_upi_id", ""),
            "amount": amount,
            "risk_score": risk_score_100,
            "risk_level": risk_level,
            "action": action,
            "action_title": action_title,
            "action_message": action_message,
            "reasons": dedup_reasons[:4],
            "models_breakdown": {
                "iqr": {
                    "score": round(iqr_score * 100, 1),
                    "is_anomaly": iqr_res["is_anomaly"],
                    "reasons": iqr_res["reasons"][:2]
                },
                "isolation_forest": {
                    "score": round(iso_score * 100, 1),
                    "is_anomaly": iso_res["is_anomaly"],
                    "reasons": iso_res["reasons"][:2]
                },
                "timeseries": {
                    "score": round(ts_score * 100, 1),
                    "is_anomaly": ts_res["is_anomaly"],
                    "reasons": ts_res["reasons"][:2]
                }
            },
            "inference_time_ms": round(inference_time_ms, 2)
        }

    def evaluate_dataset(self, df: pd.DataFrame) -> dict:
        """Benchmarks IQR, Isolation Forest, Time-Series, and Ensemble on the dataset."""
        y_true = df["is_fraud"].values
        
        # Batch predictions
        iqr_scores = self.iqr_model.predict_batch(df)
        iso_scores = self.iso_model.predict_batch(df)
        ts_scores = self.ts_model.predict_batch(df)

        # Vectorized security guardrails
        speeds = df.get("travel_speed_kmh", pd.Series(np.zeros(len(df)))).astype(float).values
        dists = df.get("distance_from_last_txn_km", pd.Series(np.zeros(len(df)))).astype(float).values
        pin_fails = df.get("failed_pin_attempts", pd.Series(np.zeros(len(df)))).astype(int).values
        trusts = df.get("device_trust_score", pd.Series(np.ones(len(df)) * 0.9)).astype(float).values
        networks = df.get("network_type", pd.Series(["4G"] * len(df))).values
        txn_types = df.get("transaction_type", pd.Series(["P2P"] * len(df))).values
        amounts = df["amount"].astype(float).values

        rule_scores = np.zeros(len(df))
        rule_scores = np.where((speeds > 800.0) & (dists > 300.0), 0.95, rule_scores)
        rule_scores = np.where((pin_fails >= 2) & (networks == "VPN_SUSPICIOUS") & (amounts > 25000), np.maximum(rule_scores, 0.92), rule_scores)
        rule_scores = np.where((txn_types == "COLLECT_REQUEST") & ((trusts < 0.5) | (networks == "VPN_SUSPICIOUS")), np.maximum(rule_scores, 0.88), rule_scores)
        rule_scores = np.where((trusts < 0.2) & (amounts >= 90000), np.maximum(rule_scores, 0.95), rule_scores)

        # Weighted Ensemble
        weighted_scores = (0.45 * iso_scores) + (0.30 * ts_scores) + (0.25 * iqr_scores)
        final_ens_scores = np.maximum(weighted_scores, rule_scores)
        final_ens_scores = np.clip(final_ens_scores, 0.0, 1.0)

        iqr_preds = (iqr_scores >= 0.60).astype(int)
        iso_preds = (iso_scores >= 0.55).astype(int)
        ts_preds = (ts_scores >= 0.60).astype(int)
        ens_preds = (final_ens_scores >= 0.60).astype(int)

        def calc_metrics(y_p, y_score=None):
            prec = precision_score(y_true, y_p, zero_division=0)
            rec = recall_score(y_true, y_p, zero_division=0)
            f1 = f1_score(y_true, y_p, zero_division=0)
            acc = accuracy_score(y_true, y_p)
            auc = roc_auc_score(y_true, y_score if y_score is not None else y_p)
            return {
                "accuracy": round(float(acc) * 100, 2),
                "precision": round(float(prec) * 100, 2),
                "recall": round(float(rec) * 100, 2),
                "f1_score": round(float(f1) * 100, 2),
                "roc_auc": round(float(auc), 4)
            }

        return {
            "iqr_technique": calc_metrics(iqr_preds, iqr_scores),
            "isolation_forest": calc_metrics(iso_preds, iso_scores),
            "timeseries_velocity": calc_metrics(ts_preds, ts_scores),
            "hybrid_ensemble": calc_metrics(ens_preds, final_ens_scores),
            "dataset_summary": {
                "total_records": len(df),
                "normal_records": int((y_true == 0).sum()),
                "fraud_records": int((y_true == 1).sum()),
                "fraud_rate_pct": round(float(y_true.mean() * 100), 2)
            }
        }
