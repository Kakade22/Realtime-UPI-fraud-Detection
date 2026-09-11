/**
 * UPI SHIELD AI - Frontend Real-Time Controller & Mobile Simulation Engine
 */

let ws = null;
let isAudioEnabled = true;
let isStreamingActive = false;
let allTransactions = [];
let audioCtx = null;

// Telemetry Counters
let totalTxnCount = 10000;
let totalFraudCount = 550;
let totalBlockedCapital = 34285000;
let avgLatencyMs = 0.45;

// Scenarios Definitions
const SCENARIOS = {
  safe_chai: {
    sender_upi_id: "rahul.verma@oksbi",
    sender_name: "Rahul Verma",
    receiver_upi_id: "chaipoint.pos@paytmqr",
    receiver_name: "Chai Point Express",
    amount: 35.0,
    transaction_type: "QR_SCAN",
    device_id: "DEV_IND_00001_SM-S918B",
    location_city: "Mumbai",
    latitude: 19.0760,
    longitude: 72.8777,
    network_type: "4G",
    device_trust_score: 0.98,
    failed_pin_attempts: 0,
    sender_account_age_days: 450,
    is_new_payee: 0
  },
  impossible_travel: {
    sender_upi_id: "rahul.verma@oksbi",
    sender_name: "Rahul Verma",
    receiver_upi_id: "suspect.node992@ybl",
    receiver_name: "Unknown Overseas Conduit",
    amount: 45000.0,
    transaction_type: "P2P",
    device_id: "DEV_UNKNOWN_9981",
    location_city: "London",
    latitude: 51.5074,
    longitude: -0.1278,
    network_type: "VPN_SUSPICIOUS",
    device_trust_score: 0.15,
    failed_pin_attempts: 1,
    sender_account_age_days: 450,
    is_new_payee: 1
  },
  velocity_drain: {
    sender_upi_id: "priya.sharma@paytm",
    sender_name: "Priya Sharma",
    receiver_upi_id: "fastdrain.mule883@paytm",
    receiver_name: "Mule Conduit 09",
    amount: 60000.0,
    transaction_type: "P2P",
    device_id: "DEV_CLONED_7721",
    location_city: "Delhi",
    latitude: 28.6139,
    longitude: 77.2090,
    network_type: "VPN_SUSPICIOUS",
    device_trust_score: 0.30,
    failed_pin_attempts: 2,
    sender_account_age_days: 210,
    is_new_payee: 1
  },
  dormant_night: {
    sender_upi_id: "amit.kumar@icici",
    sender_name: "Amit Kumar",
    receiver_upi_id: "nighttransact442@apl",
    receiver_name: "Midnight Shadow Account",
    amount: 85000.0,
    transaction_type: "P2P",
    device_id: "DEV_IND_00003_RedmiNote12",
    location_city: "Bengaluru",
    latitude: 12.9716,
    longitude: 77.5946,
    network_type: "4G",
    device_trust_score: 0.65,
    failed_pin_attempts: 1,
    sender_account_age_days: 600,
    is_new_payee: 1
  },
  phishing_qr: {
    sender_upi_id: "ananya92@apl",
    sender_name: "Ananya Patel",
    receiver_upi_id: "claim.rewards.gov99@axl",
    receiver_name: "Cashback Rewards NPCI (Fake)",
    amount: 24999.0,
    transaction_type: "COLLECT_REQUEST",
    device_id: "DEV_PHISH_3321",
    location_city: "Jaipur",
    latitude: 26.9124,
    longitude: 75.7873,
    network_type: "VPN_SUSPICIOUS",
    device_trust_score: 0.20,
    failed_pin_attempts: 0,
    sender_account_age_days: 120,
    is_new_payee: 1
  }
};

// ==========================================
// Initialization
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initWebSocket();
  fetchModelMetrics();
  fetchDataSummary();
});

function initClock() {
  function updateTime() {
    const now = new Date();
    const hrs = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    const el = document.getElementById("phone-clock");
    if (el) el.textContent = `${hrs}:${mins}`;
  }
  updateTime();
  setInterval(updateTime, 10000);
}

// ==========================================
// WebSocket Connection
// ==========================================
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live-feed`;

  const statusPill = document.getElementById("ws-status-pill");
  const statusText = document.getElementById("ws-status-text");

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    statusPill.className = "live-status-pill online";
    statusText.textContent = "Live Node Connected";
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "INITIAL_HISTORY") {
        renderInitialHistory(msg.data);
      } else if (msg.type === "TRANSACTION_EVALUATED") {
        handleIncomingTransaction(msg.data);
      }
    } catch (e) {
      console.error("WebSocket message parse error:", e);
    }
  };

  ws.onclose = () => {
    statusPill.className = "live-status-pill";
    statusText.textContent = "Reconnecting...";
    setTimeout(initWebSocket, 2500);
  };

  ws.onerror = () => {
    statusPill.className = "live-status-pill";
    statusText.textContent = "Connection Error";
  };
}

// ==========================================
// Process & Render Transactions
// ==========================================
function handleIncomingTransaction(eventData) {
  allTransactions.unshift(eventData);
  if (allTransactions.length > 200) allTransactions.pop();

  const txn = eventData.transaction;
  const evalRes = eventData.evaluation;

  // Update KPI counters
  totalTxnCount++;
  document.getElementById("kpi-total-txns").textContent = totalTxnCount.toLocaleString('en-IN');

  if (evalRes.risk_level === "FRAUD") {
    totalFraudCount++;
    totalBlockedCapital += txn.amount;
    document.getElementById("kpi-frauds-caught").textContent = totalFraudCount.toLocaleString('en-IN');
    document.getElementById("kpi-capital-blocked").textContent = `₹ ${Math.round(totalBlockedCapital).toLocaleString('en-IN')}`;
    const rate = ((totalFraudCount / totalTxnCount) * 100).toFixed(2);
    document.getElementById("kpi-fraud-rate").innerHTML = `<span class="badge-red">${rate}%</span> of total volume`;
  }

  // Latency SLA
  if (evalRes.inference_time_ms) {
    avgLatencyMs = (avgLatencyMs * 0.8) + (evalRes.inference_time_ms * 0.2);
    document.getElementById("top-latency").textContent = `${avgLatencyMs.toFixed(2)} ms`;
  }

  // Prepend row to table
  const tbody = document.getElementById("live-feed-body");
  const row = createTransactionRow(eventData);
  tbody.insertBefore(row, tbody.firstChild);

  // Keep max 35 rows visible in DOM for crisp performance
  while (tbody.children.length > 35) {
    tbody.removeChild(tbody.lastChild);
  }

  // Update Simulated Mobile Device
  updateMobileSimulation(eventData);
}

function renderInitialHistory(historyList) {
  const tbody = document.getElementById("live-feed-body");
  tbody.innerHTML = "";
  if (!historyList || historyList.length === 0) return;

  historyList.forEach(eventData => {
    allTransactions.push(eventData);
    const row = createTransactionRow(eventData);
    tbody.appendChild(row);
  });

  document.getElementById("stream-counter-tag").textContent = `${historyList.length} loaded`;
}

function createTransactionRow(eventData) {
  const txn = eventData.transaction;
  const res = eventData.evaluation;

  const tr = document.createElement("tr");
  if (res.risk_level === "FRAUD") tr.className = "row-fraud";
  else if (res.risk_level === "CAUTION") tr.className = "row-caution";

  const timeStr = eventData.server_time || txn.timestamp?.split(" ")[1] || "Just now";
  const txnShortId = txn.transaction_id.replace("TXN_UPI_", "TXN_").replace("TXN_STREAM_", "STR_");

  let badgeClass = "safe";
  let badgeIcon = "✔";
  if (res.risk_level === "FRAUD") {
    badgeClass = "fraud";
    badgeIcon = "🛑";
  } else if (res.risk_level === "CAUTION") {
    badgeClass = "caution";
    badgeIcon = "⚠️";
  }

  tr.innerHTML = `
    <td class="cell-time">
      <strong>${txnShortId}</strong>
      <span>${timeStr}</span>
    </td>
    <td>
      <div><strong>${txn.sender_name || 'User'}</strong></div>
      <small style="color:var(--text-muted);font-family:var(--font-mono);font-size:11px;">${txn.sender_upi_id}</small>
    </td>
    <td>
      <div><strong>${txn.receiver_name || 'Recipient'}</strong></div>
      <small style="color:var(--text-muted);font-family:var(--font-mono);font-size:11px;">${txn.receiver_upi_id}</small>
    </td>
    <td class="cell-amount">₹ ${parseFloat(txn.amount).toLocaleString('en-IN', { minimumFractionDigits: 2 })}</td>
    <td><span style="font-size:11px;font-weight:600;color:var(--text-secondary);">${txn.transaction_type}</span></td>
    <td><span style="font-size:11px;color:var(--text-muted);">${txn.location_city}</span></td>
    <td>
      <span class="badge-risk ${badgeClass}">
        <span>${badgeIcon}</span> ${res.risk_score}/100
      </span>
    </td>
    <td>
      <span style="font-size:11px;font-weight:700;color:${res.risk_level === 'FRAUD' ? 'var(--accent-crimson)' : (res.risk_level === 'CAUTION' ? 'var(--accent-amber)' : 'var(--accent-emerald)')};">
        ${res.action}
      </span>
    </td>
    <td>
      <button class="btn-inspect" onclick="inspectTransaction('${txn.transaction_id}')">Inspect</button>
    </td>
  `;

  return tr;
}

// ==========================================
// Simulated Mobile UPI Client Engine
// ==========================================
function updateMobileSimulation(eventData) {
  const txn = eventData.transaction;
  const res = eventData.evaluation;

  const phoneIdle = document.getElementById("phone-idle-state");
  const phonePayment = document.getElementById("phone-payment-state");
  const phoneDevice = document.getElementById("mobile-device-shell");

  const simPayName = document.getElementById("sim-pay-name");
  const simPayUpi = document.getElementById("sim-pay-upi");
  const simPayAmt = document.getElementById("sim-pay-amt");
  const simPayAvatar = document.getElementById("sim-pay-avatar");

  const warningCard = document.getElementById("sim-instant-warning-card");
  const normalApprovedCard = document.getElementById("sim-normal-approved-card");
  const cautionCard = document.getElementById("sim-caution-card");

  // Show Payment Screen
  phoneIdle.classList.add("hidden");
  phonePayment.classList.remove("hidden");

  simPayName.textContent = txn.receiver_name || "Merchant / Payee";
  simPayUpi.textContent = txn.receiver_upi_id;
  simPayAmt.textContent = `₹ ${parseFloat(txn.amount).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;

  // Set Avatar emoji based on type
  if (txn.transaction_type === "QR_SCAN") simPayAvatar.textContent = "📱";
  else if (txn.transaction_type === "P2M") simPayAvatar.textContent = "🏪";
  else if (txn.transaction_type === "COLLECT_REQUEST") simPayAvatar.textContent = "🎣";
  else simPayAvatar.textContent = "👤";

  // Reset Sub-cards
  warningCard.classList.add("hidden");
  normalApprovedCard.classList.add("hidden");
  cautionCard.classList.add("hidden");
  phoneDevice.classList.remove("vibrate-alert");

  if (res.risk_level === "FRAUD") {
    // 🛑 CRITICAL FRAUD BLOCKED
    phoneDevice.classList.add("vibrate-alert");
    playFraudSiren();

    warningCard.classList.remove("hidden");
    document.getElementById("sim-warning-title").textContent = res.action_title || "🛑 FRAUD INTERCEPTED";
    document.getElementById("sim-warning-msg").textContent = res.action_message;

    const reasonsList = document.getElementById("sim-reasons-list");
    reasonsList.innerHTML = "";
    (res.reasons || ["Anomaly detected across transaction features"]).forEach(reason => {
      const item = document.createElement("div");
      item.className = "reason-bullet-item";
      item.innerHTML = `<span>⚠️</span><span>${reason}</span>`;
      reasonsList.appendChild(item);
    });

    document.getElementById("sim-risk-val").textContent = `${res.risk_score}/100 (CRITICAL)`;
    document.getElementById("sim-meter-fill").style.width = `${res.risk_score}%`;

  } else if (res.risk_level === "CAUTION") {
    // ⚠️ CAUTION STEP-UP
    cautionCard.classList.remove("hidden");
    document.getElementById("sim-caution-msg").textContent = res.reasons?.[0] || "Unusual payment parameters detected. Please verify before entering UPI PIN.";
    playSoftChime();

  } else {
    // ✅ APPROVED
    normalApprovedCard.classList.remove("hidden");
  }
}

function acknowledgeMobileAlert() {
  const phoneDevice = document.getElementById("mobile-device-shell");
  phoneDevice.classList.remove("vibrate-alert");
  alert("🔒 Security Action Executed: Payee UPI ID reported to NPCI & blocked on your device.");
  resetMobileScreen();
}

function dismissMobileAlert() {
  const phoneDevice = document.getElementById("mobile-device-shell");
  phoneDevice.classList.remove("vibrate-alert");
  resetMobileScreen();
}

function resetMobileScreen() {
  document.getElementById("phone-payment-state").classList.add("hidden");
  document.getElementById("phone-idle-state").classList.remove("hidden");
  document.getElementById("mobile-device-shell").classList.remove("vibrate-alert");
}

// ==========================================
// Web Audio API Synthesizer (Instant Warning Sounds)
// ==========================================
function getAudioContext() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) audioCtx = new AudioContext();
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function playFraudSiren() {
  if (!isAudioEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(800, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(350, ctx.currentTime + 0.35);

    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.35);
  } catch (e) {
    console.warn("Audio synthesizer error:", e);
  }
}

function playSoftChime() {
  if (!isAudioEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(580, ctx.currentTime);
    osc.frequency.setValueAtTime(880, ctx.currentTime + 0.1);

    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (e) {}
}

function toggleAudioAlerts() {
  isAudioEnabled = !isAudioEnabled;
  const icon = document.getElementById("sound-icon");
  if (isAudioEnabled) {
    icon.textContent = "🔊";
    playSoftChime();
  } else {
    icon.textContent = "🔇";
  }
}

// ==========================================
// 1-Click Scenario Injector & Custom Form
// ==========================================
async function injectScenario(scenarioKey) {
  const scenarioData = SCENARIOS[scenarioKey];
  if (!scenarioData) return;

  // Visual button feedback
  try {
    const response = await fetch("/api/v1/transaction/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scenarioData)
    });
    const result = await response.json();
    inspectTransaction(result.transaction.transaction_id);
  } catch (err) {
    console.error("Scenario injection error:", err);
    alert("Error sending transaction to server.");
  }
}

function openCustomTxnModal() {
  document.getElementById("custom-txn-modal").classList.remove("hidden");
}

function closeCustomTxnModal() {
  document.getElementById("custom-txn-modal").classList.add("hidden");
}

async function handleCustomTxnSubmit(event) {
  event.preventDefault();
  const payload = {
    sender_upi_id: document.getElementById("cust-sender-upi").value,
    sender_name: document.getElementById("cust-sender-name").value,
    receiver_upi_id: document.getElementById("cust-receiver-upi").value,
    receiver_name: document.getElementById("cust-receiver-name").value,
    amount: parseFloat(document.getElementById("cust-amount").value),
    transaction_type: document.getElementById("cust-type").value,
    location_city: document.getElementById("cust-city").value,
    network_type: document.getElementById("cust-network").value,
    device_trust_score: parseFloat(document.getElementById("cust-trust").value),
    failed_pin_attempts: parseInt(document.getElementById("cust-pin-fails").value)
  };

  closeCustomTxnModal();

  try {
    const res = await fetch("/api/v1/transaction/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    inspectTransaction(data.transaction.transaction_id);
  } catch (e) {
    console.error("Custom txn submit failed:", e);
  }
}

// ==========================================
// Live Stream Controller
// ==========================================
async function toggleStream() {
  const btn = document.getElementById("btn-toggle-stream");
  const icon = document.getElementById("stream-btn-icon");
  const text = document.getElementById("stream-btn-text");

  if (!isStreamingActive) {
    const speed = parseFloat(document.getElementById("stream-speed-slider").value) / 1000.0;
    const res = await fetch(`/api/v1/stream/start?delay=${speed}`, { method: "POST" });
    if (res.ok) {
      isStreamingActive = true;
      btn.className = "btn btn-stream active";
      icon.textContent = "⏸";
      text.textContent = "Pause Live Stream";
      document.getElementById("kpi-stream-rate").textContent = `Stream Active (${speed}s)`;
    }
  } else {
    const res = await fetch("/api/v1/stream/stop", { method: "POST" });
    if (res.ok) {
      isStreamingActive = false;
      btn.className = "btn btn-stream";
      icon.textContent = "▶";
      text.textContent = "Start Live Stream";
      document.getElementById("kpi-stream-rate").textContent = "Stream Paused";
    }
  }
}

async function updateStreamSpeed(val) {
  const speedSec = (parseFloat(val) / 1000.0).toFixed(1);
  document.getElementById("speed-val-display").textContent = `${speedSec}s`;
  if (isStreamingActive) {
    await fetch(`/api/v1/stream/start?delay=${speedSec}`, { method: "POST" });
    document.getElementById("kpi-stream-rate").textContent = `Stream Active (${speedSec}s)`;
  }
}

// ==========================================
// Tabs & Explainability Inspection Hub
// ==========================================
function switchModelTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.add("hidden"));

  if (tabId === "models-benchmark") {
    document.getElementById("tab-models-benchmark").classList.remove("hidden");
    event?.currentTarget?.classList.add("active");
  } else if (tabId === "models-explain") {
    document.getElementById("tab-models-explain").classList.remove("hidden");
    event?.currentTarget?.classList.add("active");
  } else if (tabId === "dataset-view") {
    document.getElementById("tab-dataset-view").classList.remove("hidden");
    event?.currentTarget?.classList.add("active");
    loadDatasetExplorer();
  }
}

function inspectTransaction(txnId) {
  const found = allTransactions.find(t => t.transaction.transaction_id === txnId);
  if (!found) return;

  // Switch to Explainability Tab
  switchModelTab("models-explain");
  const tabBtns = document.querySelectorAll(".tab-selectors .tab-btn");
  tabBtns.forEach(b => {
    if (b.textContent.includes("Explainability")) b.classList.add("active");
    else b.classList.remove("active");
  });

  const txn = found.transaction;
  const ev = found.evaluation;
  const m = ev.models_breakdown;

  const box = document.getElementById("inspect-detail-box");
  box.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;border-bottom:1px solid var(--border-subtle);padding-bottom:12px;">
      <div>
        <h3 style="font-size:16px;color:var(--accent-cyan);">🔬 Explainability Deep Dive: ${txn.transaction_id}</h3>
        <p style="font-size:12px;color:var(--text-muted);">${txn.sender_upi_id} ➔ ${txn.receiver_upi_id} (₹ ${txn.amount.toLocaleString('en-IN')})</p>
      </div>
      <div class="badge-risk ${ev.risk_level.toLowerCase()}" style="font-size:13px;padding:6px 14px;">
        Risk: ${ev.risk_score}/100 • ${ev.action}
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:14px;margin-bottom:18px;">
      <div style="background:rgba(255,255,255,0.03);padding:14px;border-radius:10px;border-left:3px solid #94a3b8;">
        <h5 style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">1. Simple IQR Technique</h5>
        <h3 style="font-size:20px;font-family:var(--font-mono);">${m.iqr.score}%</h3>
        <small style="color:var(--text-muted);display:block;margin-top:4px;">${m.iqr.reasons[0] || 'Within normal IQR boundaries'}</small>
      </div>

      <div style="background:rgba(255,255,255,0.03);padding:14px;border-radius:10px;border-left:3px solid var(--accent-cyan);">
        <h5 style="font-size:12px;color:var(--accent-cyan);margin-bottom:4px;">2. Isolation Forest (13D)</h5>
        <h3 style="font-size:20px;font-family:var(--font-mono);">${m.isolation_forest.score}%</h3>
        <small style="color:var(--text-muted);display:block;margin-top:4px;">${m.isolation_forest.reasons[0] || 'Normal multidimensional cluster'}</small>
      </div>

      <div style="background:rgba(255,255,255,0.03);padding:14px;border-radius:10px;border-left:3px solid var(--accent-amber);">
        <h5 style="font-size:12px;color:var(--accent-amber);margin-bottom:4px;">3. Time Series & Velocity</h5>
        <h3 style="font-size:20px;font-family:var(--font-mono);">${m.timeseries.score}%</h3>
        <small style="color:var(--text-muted);display:block;margin-top:4px;">${m.timeseries.reasons[0] || 'Steady Poisson arrival interval'}</small>
      </div>
    </div>

    <div style="background:rgba(0,0,0,0.3);padding:14px;border-radius:10px;">
      <h5 style="font-size:12px;color:var(--text-secondary);margin-bottom:8px;">Decision Rationale & Triggered Guardrails:</h5>
      <ul style="padding-left:18px;font-size:12px;color:var(--text-primary);line-height:1.7;">
        ${ev.reasons.map(r => `<li><strong>${r}</strong></li>`).join('')}
      </ul>
    </div>
  `;
}

// ==========================================
// Model Metrics & Dataset Explorer
// ==========================================
async function fetchModelMetrics() {
  try {
    const res = await fetch("/api/v1/models/metrics");
    if (!res.ok) return;
    const data = await res.json();

    // Populate IQR metrics
    if (data.iqr_technique) {
      document.getElementById("iqr-acc").textContent = `${data.iqr_technique.accuracy}%`;
      document.getElementById("iqr-rec").textContent = `${data.iqr_technique.recall}%`;
      document.getElementById("iqr-auc").textContent = data.iqr_technique.roc_auc;
    }

    // Populate Isolation Forest metrics
    if (data.isolation_forest) {
      document.getElementById("iso-acc").textContent = `${data.isolation_forest.accuracy}%`;
      document.getElementById("iso-prec").textContent = `${data.isolation_forest.precision}%`;
      document.getElementById("iso-auc").textContent = data.isolation_forest.roc_auc;
    }

    // Populate Time Series metrics
    if (data.timeseries_velocity) {
      document.getElementById("ts-acc").textContent = `${data.timeseries_velocity.accuracy}%`;
      document.getElementById("ts-prec").textContent = `${data.timeseries_velocity.precision}%`;
    }

    // Populate Master Ensemble metrics
    if (data.hybrid_ensemble) {
      document.getElementById("ens-acc").textContent = `${data.hybrid_ensemble.accuracy}%`;
      document.getElementById("ens-auc").textContent = data.hybrid_ensemble.roc_auc;
      document.getElementById("kpi-model-auc").innerHTML = `${data.hybrid_ensemble.accuracy}% <span class="sub-auc">(${data.hybrid_ensemble.roc_auc})</span>`;
    }
  } catch (e) {
    console.warn("Metrics fetch error:", e);
  }
}

async function fetchDataSummary() {
  try {
    const res = await fetch("/api/v1/data/summary");
    if (!res.ok) return;
    const summary = await res.json();
    if (summary.total_records) {
      totalTxnCount = summary.total_records;
      totalFraudCount = summary.total_frauds;
      document.getElementById("kpi-total-txns").textContent = totalTxnCount.toLocaleString('en-IN');
      document.getElementById("kpi-frauds-caught").textContent = totalFraudCount.toLocaleString('en-IN');
      document.getElementById("kpi-fraud-rate").innerHTML = `<span class="badge-red">${summary.fraud_rate_pct}%</span> of total volume`;
    }
  } catch (e) {}
}

let datasetCache = [];
async function loadDatasetExplorer() {
  const tbody = document.getElementById("dataset-explorer-body");
  if (datasetCache.length > 0) return;

  tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--text-muted);">Loading 10,000 synthetic transaction records...</td></tr>`;

  try {
    // Read recent transactions or dataset sample
    const res = await fetch("/api/v1/transactions/recent?limit=100");
    const data = await res.json();
    datasetCache = data.map(d => d.transaction);
    renderDatasetTable(datasetCache);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--accent-crimson);">Failed to load dataset records.</td></tr>`;
  }
}

function renderDatasetTable(items) {
  const tbody = document.getElementById("dataset-explorer-body");
  tbody.innerHTML = "";

  if (items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--text-muted);">No matching transactions found.</td></tr>`;
    return;
  }

  items.slice(0, 50).forEach(t => {
    const tr = document.createElement("tr");
    const isFraud = t.is_fraud === 1 || t.fraud_type !== "NONE";
    if (isFraud) tr.className = "row-fraud";

    tr.innerHTML = `
      <td style="font-family:var(--font-mono);font-size:11px;">${t.transaction_id}</td>
      <td style="font-size:11px;color:var(--text-muted);">${t.timestamp}</td>
      <td style="font-family:var(--font-mono);font-size:11px;">${t.sender_upi_id}</td>
      <td class="cell-amount">₹ ${parseFloat(t.amount).toLocaleString('en-IN')}</td>
      <td>${t.location_city}</td>
      <td style="font-family:var(--font-mono);font-size:11px;">${t.travel_speed_kmh || 0} km/h</td>
      <td>${t.device_trust_score || 0.9}</td>
      <td>
        <span class="badge-risk ${isFraud ? 'fraud' : 'safe'}">
          ${t.fraud_type || (isFraud ? 'FRAUD' : 'NORMAL')}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterDatasetTable() {
  const query = document.getElementById("dataset-search").value.toLowerCase();
  const filterType = document.getElementById("dataset-fraud-filter").value;

  const filtered = datasetCache.filter(t => {
    const matchesQuery = !query || 
      t.sender_upi_id.toLowerCase().includes(query) || 
      t.location_city.toLowerCase().includes(query) ||
      (t.sender_name && t.sender_name.toLowerCase().includes(query));

    let matchesType = true;
    if (filterType === "FRAUD_ONLY") matchesType = (t.is_fraud === 1 || t.fraud_type !== "NONE");
    else if (filterType !== "ALL") matchesType = (t.fraud_type === filterType);

    return matchesQuery && matchesType;
  });

  renderDatasetTable(filtered);
}
