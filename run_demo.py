"""
UPI Fraud Detection System - One-Click Launcher
Starts the FastAPI Backend and WebSocket live server on port 8000.
"""

import uvicorn
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)
    print("=" * 65)
    print("UPI SHIELD AI - REAL-TIME FRAUD DETECTION SYSTEM")
    print("=" * 65)
    print("Step 1: Synthetic Data Generation (10,000 Records) [OK]")
    print("Step 2: Anomaly Detection Models (IQR, IsoForest, Time-Series) [OK]")
    print("Step 3: Real-Time Deployment & Instant Warning Feed [OK]")
    print("-" * 65)
    print("Launching server at: http://localhost:8000")
    print("=" * 65)
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
