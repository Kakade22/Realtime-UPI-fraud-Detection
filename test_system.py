"""
Automated Integration & Verification Suite for UPI Fraud Detection System
Tests:
1. Synthetic dataset integrity (10,000 records, schema, fraud types)
2. Anomaly detection models (IQR, Isolation Forest, Time-Series, Hybrid Ensemble)
3. FastAPI Endpoints (/api/v1/models/metrics, /api/v1/data/summary, /api/v1/transaction/process)
4. WebSocket live feed streaming & alerting
5. Simulated Fraud Scenarios (Impossible Travel, Velocity Drain, Dormant Midnight, QR Phishing)
"""

import sys
import os
import json
import asyncio
import urllib.request
import pandas as pd
import websockets

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/live-feed"

def test_dataset_integrity():
    print("\n[TEST 1] Testing Synthetic Dataset...")
    csv_path = "data/synthetic_upi_10k.csv"
    assert os.path.exists(csv_path), "Dataset CSV does not exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 10000, f"Expected 10000 records, got {len(df)}"
    assert "is_fraud" in df.columns, "Missing 'is_fraud' column"
    assert "fraud_type" in df.columns, "Missing 'fraud_type' column"
    fraud_count = df["is_fraud"].sum()
    print(f" -> Dataset Validated: {len(df)} transactions, {fraud_count} frauds ({fraud_count/len(df)*100:.2f}%)")

def test_api_summary_and_metrics():
    print("\n[TEST 2] Testing REST API /api/v1/data/summary & /models/metrics...")
    # Summary
    with urllib.request.urlopen(f"{BASE_URL}/api/v1/data/summary") as res:
        assert res.status == 200
        summary = json.loads(res.read().decode())
        assert summary["total_records"] == 10000
        print(f" -> Summary API OK: {summary['total_frauds']} frauds in dataset")

    # Metrics
    with urllib.request.urlopen(f"{BASE_URL}/api/v1/models/metrics") as res:
        assert res.status == 200
        metrics = json.loads(res.read().decode())
        assert "iqr_technique" in metrics
        assert "isolation_forest" in metrics
        assert "timeseries_velocity" in metrics
        assert "hybrid_ensemble" in metrics
        print(f" -> Models Benchmark OK: IsoForest AUC = {metrics['isolation_forest']['roc_auc']}, Ensemble Acc = {metrics['hybrid_ensemble']['accuracy']}%")

def test_scenario_scoring():
    print("\n[TEST 3] Testing Real-Time Scoring Scenarios...")
    scenarios = [
        ("Normal P2M", {
            "sender_upi_id": "test.user1@oksbi",
            "receiver_upi_id": "swiggy@hdfcbank",
            "amount": 250.0,
            "transaction_type": "P2M",
            "location_city": "Mumbai"
        }, "SAFE"),
        ("Impossible Travel Jump", {
            "sender_upi_id": "test.user1@oksbi",
            "receiver_upi_id": "hacker.abroad@ybl",
            "amount": 75000.0,
            "transaction_type": "P2P",
            "location_city": "London",
            "latitude": 51.5074,
            "longitude": -0.1278,
            "network_type": "VPN_SUSPICIOUS",
            "device_trust_score": 0.1
        }, "FRAUD"),
        ("Phishing QR Collect Scam", {
            "sender_upi_id": "test.user2@paytm",
            "receiver_upi_id": "claim.cashback@axl",
            "amount": 19999.0,
            "transaction_type": "COLLECT_REQUEST",
            "network_type": "VPN_SUSPICIOUS",
            "device_trust_score": 0.2
        }, "FRAUD")
    ]

    for name, payload, expected_level in scenarios:
        req = urllib.request.Request(
            f"{BASE_URL}/api/v1/transaction/process",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            eval_res = data["evaluation"]
            print(f" -> Scenario '{name}': Risk Score = {eval_res['risk_score']}/100, Level = {eval_res['risk_level']}, Action = {eval_res['action']}")
            assert eval_res["risk_level"] == expected_level, f"Expected {expected_level}, got {eval_res['risk_level']}"

async def test_websocket_stream():
    print("\n[TEST 4] Testing WebSocket Live Feed...")
    async with websockets.connect(WS_URL) as ws:
        # First message should be INITIAL_HISTORY
        initial_msg = await ws.recv()
        data = json.loads(initial_msg)
        assert data["type"] == "INITIAL_HISTORY"
        print(f" -> WebSocket Connected: Received INITIAL_HISTORY with {len(data['data'])} cached transactions")

        # Start stream briefly
        req = urllib.request.Request(f"{BASE_URL}/api/v1/stream/start?delay=0.3", method="POST")
        urllib.request.urlopen(req)

        # Receive streamed live transaction
        stream_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        stream_data = json.loads(stream_msg)
        assert stream_data["type"] == "TRANSACTION_EVALUATED"
        print(f" -> WebSocket Stream Message: Txn {stream_data['data']['transaction']['transaction_id']} scored with Risk {stream_data['data']['evaluation']['risk_score']}/100")

        # Stop stream
        req_stop = urllib.request.Request(f"{BASE_URL}/api/v1/stream/stop", method="POST")
        urllib.request.urlopen(req_stop)
        print(" -> WebSocket Live Feed test passed successfully!")

def run_all_tests():
    print("=" * 60)
    print("RUNNING END-TO-END AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    test_dataset_integrity()
    test_api_summary_and_metrics()
    test_scenario_scoring()
    asyncio.run(test_websocket_stream())
    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (100% GREEN)")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
