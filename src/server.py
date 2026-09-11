"""
FastAPI Server & Real-Time UPI Fraud Detection Deployment
Provides:
1. Real-Time Transaction Scoring & Fraud Flagging API (/api/v1/transaction/process)
2. Live WebSocket Feed (/ws/live-feed) for Instant Warning Broadcasting
3. Automated Real-Time Stream Simulator (/api/v1/stream/start, /stop)
4. Model Metrics & Data Summary APIs (/api/v1/models/metrics, /api/v1/data/summary)
5. Static Frontend Asset Hosting
"""

import os
import sys
import json
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.state_engine import StateEngine
from src.models.ensemble_engine import UPIFraudEnsembleEngine

app = FastAPI(
    title="UPI Fraud Detection & Instant Alert System",
    description="Real-Time Anomaly Detection using IQR, Isolation Forest, and Time-Series Sequential Models for Indian UPI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engines & State
state_engine = StateEngine()
ensemble_engine = UPIFraudEnsembleEngine()
recent_transactions: List[Dict[str, Any]] = []
stream_task: Optional[asyncio.Task] = None
is_streaming = False
stream_delay_seconds = 1.2

# Synthetic Dataset Reference for stream simulation
synthetic_df = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Input Pydantic Model
class UPITransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    timestamp: Optional[str] = None
    sender_upi_id: str = Field(..., example="rahul.verma@oksbi")
    sender_name: Optional[str] = Field("Rahul Verma", example="Rahul Verma")
    receiver_upi_id: str = Field(..., example="swiggy@hdfcbank")
    receiver_name: Optional[str] = Field("Swiggy Orders", example="Swiggy Orders")
    amount: float = Field(..., gt=0, example=450.0)
    transaction_type: str = Field("P2M", example="P2M")
    device_id: Optional[str] = Field(None, example="DEV_IND_0001_SM-S918B")
    location_city: str = Field("Mumbai", example="Mumbai")
    latitude: Optional[float] = Field(19.0760, example=19.0760)
    longitude: Optional[float] = Field(72.8777, example=72.8777)
    network_type: str = Field("4G", example="4G")
    device_trust_score: float = Field(0.95, ge=0.0, le=1.0, example=0.95)
    failed_pin_attempts: int = Field(0, ge=0, example=0)
    sender_account_age_days: int = Field(365, ge=1, example=365)
    is_new_payee: int = Field(0, ge=0, le=1, example=0)

@app.on_event("startup")
async def startup_event():
    global synthetic_df, ensemble_engine
    csv_path = os.path.join(BASE_DIR, "data", "synthetic_upi_10k.csv")
    if not os.path.exists(csv_path):
        print("Dataset not found. Generating 10,000 synthetic transactions...")
        from src.data_generator import generate_synthetic_upi_dataset
        synthetic_df = generate_synthetic_upi_dataset(10000)
    else:
        print(f"Loading synthetic dataset from {csv_path}...")
        synthetic_df = pd.read_csv(csv_path)

    print("Fitting models (Simple IQR, Isolation Forest, Time-Series)...")
    ensemble_engine.fit(synthetic_df)
    print("UPI Fraud Detection Engines ready for real-time inference!")

@app.post("/api/v1/transaction/process")
async def process_transaction(txn: UPITransactionRequest):
    """
    Process an incoming real-time UPI transaction:
    1. Updates real-time user sliding window & velocity state
    2. Runs IQR, Isolation Forest, and Time-Series detectors
    3. Evaluates security guardrails & ensembles risk score
    4. Broadcasts alert to WebSocket clients if fraud or caution
    """
    txn_dict = txn.model_dump()
    if not txn_dict.get("transaction_id"):
        txn_dict["transaction_id"] = f"TXN_RT_{int(datetime.now().timestamp() * 1000)}"
    if not txn_dict.get("timestamp"):
        txn_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Stateful feature engineering
    enriched_txn = state_engine.process_transaction(txn_dict)

    # Anomaly detection & scoring
    result = ensemble_engine.predict_single(enriched_txn)
    
    # Merge payload info for frontend
    full_event = {
        "transaction": enriched_txn,
        "evaluation": result,
        "server_time": datetime.now().strftime("%H:%M:%S.%f")[:-3]
    }

    # Keep in memory history (max 200 items)
    recent_transactions.insert(0, full_event)
    if len(recent_transactions) > 200:
        recent_transactions.pop()

    # Broadcast via WebSocket
    await manager.broadcast({
        "type": "TRANSACTION_EVALUATED",
        "data": full_event
    })

    return full_event

@app.get("/api/v1/models/metrics")
async def get_model_metrics():
    """Returns comparative metrics for IQR, Isolation Forest, Time Series, and Hybrid Ensemble."""
    return ensemble_engine.evaluation_metrics

@app.get("/api/v1/transactions/recent")
async def get_recent_transactions(limit: int = 50):
    """Returns recently processed transactions."""
    return recent_transactions[:limit]

@app.get("/api/v1/data/summary")
async def get_data_summary():
    """Returns dataset summary statistics and fraud distribution."""
    if synthetic_df is None:
        return {"status": "Dataset not loaded yet"}

    fraud_counts = synthetic_df["fraud_type"].value_counts().to_dict()
    txn_type_counts = synthetic_df["transaction_type"].value_counts().to_dict()
    city_counts = synthetic_df["location_city"].value_counts().head(10).to_dict()

    return {
        "total_records": len(synthetic_df),
        "total_frauds": int(synthetic_df["is_fraud"].sum()),
        "fraud_rate_pct": round(float(synthetic_df["is_fraud"].mean() * 100), 2),
        "amount_stats": {
            "min": float(synthetic_df["amount"].min()),
            "max": float(synthetic_df["amount"].max()),
            "mean": round(float(synthetic_df["amount"].mean()), 2),
            "median": round(float(synthetic_df["amount"].median()), 2),
            "p95": round(float(synthetic_df["amount"].quantile(0.95)), 2)
        },
        "fraud_distribution": fraud_counts,
        "transaction_type_distribution": txn_type_counts,
        "top_cities": city_counts
    }

# Live Stream Simulator Worker
async def transaction_stream_worker():
    global is_streaming, synthetic_df
    print("Live UPI Transaction stream started...")
    idx = 0
    df_len = len(synthetic_df) if synthetic_df is not None else 0

    while is_streaming:
        try:
            if df_len > 0:
                row = synthetic_df.iloc[idx % df_len].to_dict()
                idx += 1
                # Adjust timestamp to current real time
                row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row["transaction_id"] = f"TXN_STREAM_{int(datetime.now().timestamp() * 1000) % 10000000}"

                enriched = state_engine.process_transaction(row)
                res = ensemble_engine.predict_single(enriched)

                event = {
                    "transaction": enriched,
                    "evaluation": res,
                    "server_time": datetime.now().strftime("%H:%M:%S.%f")[:-3]
                }

                recent_transactions.insert(0, event)
                if len(recent_transactions) > 200:
                    recent_transactions.pop()

                await manager.broadcast({
                    "type": "TRANSACTION_EVALUATED",
                    "data": event
                })

            await asyncio.sleep(stream_delay_seconds)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in stream worker: {e}")
            await asyncio.sleep(1.0)

    print("Live UPI Transaction stream stopped.")

@app.post("/api/v1/stream/start")
async def start_stream(delay: float = 1.2):
    """Starts continuous background generation of real-time UPI traffic."""
    global is_streaming, stream_task, stream_delay_seconds
    stream_delay_seconds = max(0.2, min(5.0, delay))
    if not is_streaming:
        is_streaming = True
        stream_task = asyncio.create_task(transaction_stream_worker())
    return {"status": "STREAM_ACTIVE", "delay_seconds": stream_delay_seconds}

@app.post("/api/v1/stream/stop")
async def stop_stream():
    """Stops the real-time background stream."""
    global is_streaming, stream_task
    if is_streaming and stream_task:
        is_streaming = False
        stream_task.cancel()
        stream_task = None
    return {"status": "STREAM_STOPPED"}

@app.get("/api/v1/stream/status")
async def get_stream_status():
    return {
        "is_streaming": is_streaming,
        "delay_seconds": stream_delay_seconds,
        "total_recent_cached": len(recent_transactions)
    }

# WebSocket Endpoint
@app.websocket("/ws/live-feed")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Send initial recent state
    await websocket.send_json({
        "type": "INITIAL_HISTORY",
        "data": recent_transactions[:25]
    })
    try:
        while True:
            data = await websocket.receive_text()
            # Client can send ping/heartbeats or manual trigger commands
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# Mount Static Web Files
web_dir = os.path.join(BASE_DIR, "web")
os.makedirs(web_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(web_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "UPI Fraud Detection API Live. Frontend building..."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
