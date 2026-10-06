"""
app.py
-------
Streamlit dashboard for the AI-Based Vehicle Accident Detection, Severity
Prediction and Emergency/Service Center Notification System.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sensor_simulator import SensorSimulator, WINDOW_SIZE, SENSOR_COLUMNS
from prediction.predictor import get_predictor
from services.emergency_service import build_emergency_alert, find_all_nearest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Vehicle Accident Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main { background-color: #0e1117; }
.big-title {
    font-size: 2.1rem; font-weight: 800; color: #ffffff;
    padding-bottom: 0px; margin-bottom: 0px;
}
.subtitle { color: #9aa4b2; font-size: 1.0rem; margin-top: 0px; }
.status-card {
    border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}
.status-safe { background: linear-gradient(135deg,#0f3d24,#12271b); border-color:#1f6b3d; }
.status-danger { background: linear-gradient(135deg,#4a0f14,#2b0a0d); border-color:#a4222c; }
.status-warning { background: linear-gradient(135deg,#4a3a0f,#2b220a); border-color:#a48622; }
.metric-label { color:#9aa4b2; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em;}
.metric-value { color:#fff; font-size:1.6rem; font-weight:700; }
.facility-card {
    background:#161b22; border:1px solid rgba(255,255,255,0.08); border-radius:12px;
    padding:12px 16px; margin-bottom:10px;
}
.facility-name { color:#fff; font-weight:700; font-size:1.0rem;}
.facility-meta { color:#9aa4b2; font-size:0.85rem;}
.timeline-item { color:#c9d1d9; font-size:0.9rem; padding:2px 0; }
.pill {
    display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700;
    font-size:0.8rem; letter-spacing:0.04em;
}
.pill-safe { background:#1f6b3d; color:#eafff2; }
.pill-danger { background:#a4222c; color:#ffecec; }
.pill-low { background:#3a5c8a; color:#eaf2ff; }
.pill-medium { background:#8a6a1f; color:#fff6e0; }
.pill-high { background:#a4522c; color:#ffe9dc; }
.pill-critical { background:#7a1030; color:#ffdfe9; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SEVERITY_PILL_CLASS = {
    "NONE": "pill-safe", "LOW": "pill-low", "MEDIUM": "pill-medium",
    "HIGH": "pill-high", "CRITICAL": "pill-critical",
}

# ----------------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------------
def init_state():
    if "sim" not in st.session_state:
        st.session_state.sim = SensorSimulator()
    if "prediction" not in st.session_state:
        st.session_state.prediction = {
            "accident_detected": False, "accident_probability": 0.02,
            "severity": "NONE", "severity_confidence": 0.98,
        }
    if "timeline" not in st.session_state:
        st.session_state.timeline = []
    if "alert" not in st.session_state:
        st.session_state.alert = None
    if "driver_response_pending" not in st.session_state:
        st.session_state.driver_response_pending = False
    if "countdown" not in st.session_state:
        st.session_state.countdown = 30
    if "running" not in st.session_state:
        st.session_state.running = False


init_state()


@st.cache_resource(show_spinner="Loading trained CNN-LSTM models...")
def load_predictor():
    return get_predictor()


def add_timeline(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.timeline.append(f"{ts}  {msg}")
    st.session_state.timeline = st.session_state.timeline[-12:]


def run_scenario(event_type, n_samples=30):
    sim = st.session_state.sim
    add_timeline(f"Scenario triggered: {event_type.replace('_',' ').title()}")
    sim.push_event(event_type, n_samples=n_samples)

    predictor = load_predictor()
    window = sim.get_window_array()
    result = predictor.full_prediction(window)
    st.session_state.prediction = result

    if result["accident_detected"]:
        add_timeline(f"⚠ Accident Detected (confidence {result['accident_probability']*100:.1f}%)")
        add_timeline(f"Severity = {result['severity']} ({result['severity_confidence']*100:.1f}%)")

        lat, lon = sim.gps.lat, sim.gps.lon
        add_timeline("GPS Captured")
        speed_before = float(window[max(0, len(window) - 10), 0])
        acc_cols = window[:, 1:4]
        peak_g = float(np.sqrt(((acc_cols - [0, 0, 9.8]) ** 2).sum(axis=1)).max() / 9.8)
        tilt = float(window[-1, 9])

        alert = build_emergency_alert(
            vehicle_id=sim.vehicle_id,
            accident_prob=result["accident_probability"],
            severity_label=result["severity"],
            severity_confidence=result["severity_confidence"],
            speed_before_impact=speed_before,
            peak_acceleration_g=peak_g,
            tilt_angle=tilt,
            lat=lat, lon=lon,
        )
        st.session_state.alert = alert
        add_timeline("Nearest Services Located")
        st.session_state.driver_response_pending = True
        st.session_state.countdown = 30
    else:
        st.session_state.alert = None
        st.session_state.driver_response_pending = False


# ----------------------------------------------------------------------------
# SIDEBAR - SIMULATION CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛 Simulation Controls")
    st.caption("Trigger a driving scenario. The CNN-LSTM models analyze the "
               "resulting sensor window in real time.")

    if st.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state.sim.reset()
        st.session_state.prediction = {
            "accident_detected": False, "accident_probability": 0.02,
            "severity": "NONE", "severity_confidence": 0.98,
        }
        st.session_state.timeline = []
        st.session_state.alert = None
        st.session_state.driver_response_pending = False
        add_timeline("Simulation reset")
        st.rerun()

    st.markdown("#### Non-Accident Scenarios")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚗 Normal Driving", use_container_width=True):
            run_scenario("NORMAL", 30)
            st.rerun()
        if st.button("🕳 Pothole", use_container_width=True):
            run_scenario("POTHOLE", 20)
            st.rerun()
    with c2:
        if st.button("🛑 Hard Braking", use_container_width=True):
            run_scenario("HARD_BRAKING", 25)
            st.rerun()
        if st.button("↪ Sharp Turn", use_container_width=True):
            run_scenario("SHARP_TURN", 25)
            st.rerun()

    st.markdown("#### Accident Scenarios")
    if st.button("💥 Minor Accident", use_container_width=True):
        run_scenario("MINOR_ACCIDENT", 30)
        st.rerun()
    if st.button("🚨 Major Accident", use_container_width=True, type="primary"):
        run_scenario("MAJOR_ACCIDENT", 30)
        st.rerun()
    if st.button("🔃 Rollover", use_container_width=True):
        run_scenario("ROLLOVER", 30)
        st.rerun()

    st.divider()
    st.caption("This entire system is a software simulation for a college "
               "Deep Learning project. No physical sensors, hardware, or "
               "real emergency services are involved.")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="big-title">🚨 AI-Based Vehicle Accident Detection & Emergency Notification System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN-LSTM Deep Learning · Real-Time Simulated Sensors · '
            'Automatic Severity Prediction · Simulated GPS & Emergency Alerting</p>',
            unsafe_allow_html=True)
st.write("")

tab_dashboard, tab_models, tab_compare, tab_about = st.tabs(
    ["📡 Live Dashboard", "📊 Model Performance", "⚖ Threshold vs Deep Learning", "ℹ️ About the Project"]
)

# ----------------------------------------------------------------------------
# TAB 1 : LIVE DASHBOARD
# ----------------------------------------------------------------------------
with tab_dashboard:
    sim = st.session_state.sim
    pred = st.session_state.prediction
    window = sim.get_window_array()
    last_row = {col: window[-1, i] for i, col in enumerate(SENSOR_COLUMNS)}

    col_left, col_right = st.columns([1.15, 1])

    # ---- Vehicle Status Card ----
    with col_left:
        status_class = "status-danger" if pred["accident_detected"] else "status-safe"
        status_text = "🔴 ACCIDENT DETECTED" if pred["accident_detected"] else "🟢 VEHICLE SAFE"
        st.markdown(f"""
        <div class="status-card {status_class}">
            <div style="font-size:1.3rem;font-weight:800;color:#fff;">{status_text}</div>
            <div style="display:flex; gap:32px; margin-top:10px; flex-wrap:wrap;">
                <div><div class="metric-label">Speed</div><div class="metric-value">{last_row['speed']:.0f} km/h</div></div>
                <div><div class="metric-label">Temperature</div><div class="metric-value">{last_row['temperature']:.1f}°C</div></div>
                <div><div class="metric-label">Vibration</div><div class="metric-value">{last_row['vibration']:.1f}</div></div>
                <div><div class="metric-label">Tilt</div><div class="metric-value">{last_row['tilt_angle']:.1f}°</div></div>
                <div><div class="metric-label">GPS</div><div class="metric-value">Active</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Live sensor graphs
        st.markdown("#### 📈 Live Sensor Graph (last {} readings)".format(WINDOW_SIZE))
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Speed (km/h)", "Acceleration (m/s²)", "Gyroscope (°/s)", "Vibration & Tilt"))

        x_axis = list(range(len(window)))
        fig.add_trace(go.Scatter(y=window[:, 0], x=x_axis, name="Speed",
                                  line=dict(color="#4fd1c5")), row=1, col=1)
        fig.add_trace(go.Scatter(y=window[:, 1], x=x_axis, name="Acc X", line=dict(color="#f6ad55")), row=1, col=2)
        fig.add_trace(go.Scatter(y=window[:, 2], x=x_axis, name="Acc Y", line=dict(color="#fc8181")), row=1, col=2)
        fig.add_trace(go.Scatter(y=window[:, 3], x=x_axis, name="Acc Z", line=dict(color="#b794f4")), row=1, col=2)
        fig.add_trace(go.Scatter(y=window[:, 4], x=x_axis, name="Gyro X", line=dict(color="#68d391")), row=2, col=1)
        fig.add_trace(go.Scatter(y=window[:, 5], x=x_axis, name="Gyro Y", line=dict(color="#63b3ed")), row=2, col=1)
        fig.add_trace(go.Scatter(y=window[:, 6], x=x_axis, name="Gyro Z", line=dict(color="#f687b3")), row=2, col=1)
        fig.add_trace(go.Scatter(y=window[:, 7], x=x_axis, name="Vibration", line=dict(color="#ecc94b")), row=2, col=2)
        fig.add_trace(go.Scatter(y=window[:, 9], x=x_axis, name="Tilt", line=dict(color="#fc8181")), row=2, col=2)

        fig.update_layout(height=460, template="plotly_dark", showlegend=True,
                           margin=dict(l=10, r=10, t=40, b=10),
                           legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # BEFORE / DURING / AFTER analysis
        st.markdown("#### 🕒 Before → During → After Analysis")
        n = len(window)
        before = window[: n // 3]
        during = window[n // 3: 2 * n // 3]
        after = window[2 * n // 3:]
        stages = st.columns(3)
        stage_data = [("BEFORE", before, "🟢"), ("DURING", during, "🟠"), ("AFTER", after, "🔴")]
        for (label, seg, icon), col in zip(stage_data, stages):
            with col:
                st.markdown(f"**{icon} {label}**")
                st.caption(f"Speed: {seg[:,0].mean():.0f} km/h")
                st.caption(f"Peak Vibration: {seg[:,7].max():.1f}")
                st.caption(f"Peak Tilt: {seg[:,9].max():.1f}°")

    # ---- AI Prediction / Severity / Alert ----
    with col_right:
        st.markdown("#### 🧠 AI Accident Detection")
        acc_prob = pred["accident_probability"]
        st.markdown(f"""
        <div class="status-card {'status-danger' if pred['accident_detected'] else 'status-safe'}">
            <div class="metric-label">Normal: {(1-acc_prob)*100:.1f}%  |  Accident: {acc_prob*100:.1f}%</div>
            <div style="font-size:1.2rem; font-weight:800; color:#fff; margin-top:6px;">
                {'🚨 ACCIDENT DETECTED' if pred['accident_detected'] else '✅ NO ACCIDENT'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(acc_prob, 0.0), 1.0))

        st.markdown("#### 🧭 Severity Prediction")
        sev = pred["severity"]
        pill_class = SEVERITY_PILL_CLASS.get(sev, "pill-safe")
        st.markdown(f"""
        <div class="status-card status-warning">
            <span class="pill {pill_class}">{sev}</span>
            <div style="margin-top:8px; color:#fff; font-weight:700;">
                Confidence: {pred['severity_confidence']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📍 Accident Location (Simulated GPS)")
        st.caption(f"Latitude: {sim.gps.lat:.4f}   |   Longitude: {sim.gps.lon:.4f}")
        map_df = pd.DataFrame({"lat": [sim.gps.lat], "lon": [sim.gps.lon]})
        st.map(map_df, size=200, color="#ff4b4b", zoom=12)

        if pred["accident_detected"] and st.session_state.alert:
            st.markdown("#### 🏥 Nearby Emergency & Service Facilities")
            nearest = st.session_state.alert["nearest_facilities"]
            icons = {"Hospital": "🏥", "Ambulance": "🚑", "Police Station": "🚓",
                     "Fire Station": "🚒", "Vehicle Service Center": "🔧", "Towing Center": "🛻"}
            for ftype, info in nearest.items():
                if info is None:
                    continue
                st.markdown(f"""
                <div class="facility-card">
                    <div class="facility-name">{icons.get(ftype,'📍')} {info['name']}</div>
                    <div class="facility-meta">{ftype} · {info['distance_km']} km away · {info['availability']}</div>
                    <div class="facility-meta">📞 {info['phone']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🚗 Driver Response Simulation")
            if st.session_state.driver_response_pending:
                st.warning("Are you safe?")
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("✅ I'M SAFE", use_container_width=True):
                        st.session_state.driver_response_pending = False
                        add_timeline("Driver responded: I'M SAFE - alert cancelled")
                        st.rerun()
                with r2:
                    if st.button("📣 SEND EMERGENCY ALERT NOW", use_container_width=True, type="primary"):
                        st.session_state.driver_response_pending = False
                        add_timeline("Driver manually triggered EMERGENCY ALERT")
                        st.rerun()
                if st.button("⏱ SIMULATE NO RESPONSE (auto-trigger alert)", use_container_width=True):
                    st.session_state.driver_response_pending = False
                    add_timeline("No driver response - EMERGENCY ALERT ACTIVATED automatically")
                    st.rerun()
            else:
                st.success("📤 Emergency Alert Generated & Sent (Simulated)")
                with st.expander("View full alert payload"):
                    st.json(st.session_state.alert)

        st.markdown("#### 🕓 Alert Timeline")
        if st.session_state.timeline:
            for item in reversed(st.session_state.timeline):
                st.markdown(f'<div class="timeline-item">• {item}</div>', unsafe_allow_html=True)
        else:
            st.caption("No events yet. Trigger a scenario from the sidebar.")

# ----------------------------------------------------------------------------
# TAB 2 : MODEL PERFORMANCE
# ----------------------------------------------------------------------------
with tab_models:
    st.markdown("### 📊 Trained Deep Learning Model Performance")
    st.caption("Metrics computed on a held-out test split, never seen during training.")

    acc_metrics_path = os.path.join(MODELS_DIR, "accident_model_metrics.json")
    sev_metrics_path = os.path.join(MODELS_DIR, "severity_model_metrics.json")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🚗 Accident Detection Model (CNN-LSTM)")
        if os.path.exists(acc_metrics_path):
            with open(acc_metrics_path) as f:
                m = json.load(f)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            k2.metric("Recall", f"{m['recall']*100:.1f}%", help="Most safety-critical metric")
            k3.metric("Precision", f"{m['precision']*100:.1f}%")
            k4.metric("ROC-AUC", f"{m['roc_auc']*100:.1f}%")
            st.caption(f"False Negatives (missed accidents on test set): {m['false_negatives']} "
                       f"out of {m['n_test']} test samples.")
        else:
            st.info("Run `python models/train_accident_model.py` to generate metrics.")

        for img_name, caption in [
            ("accident_accuracy_curve.png", "Training vs Validation Accuracy"),
            ("accident_loss_curve.png", "Training vs Validation Loss"),
            ("accident_confusion_matrix.png", "Confusion Matrix"),
            ("accident_roc_curve.png", "ROC Curve"),
        ]:
            p = os.path.join(MODELS_DIR, img_name)
            if os.path.exists(p):
                st.image(p, caption=caption, use_column_width=True)

    with colB:
        st.markdown("#### 🧭 Severity Prediction Model (CNN-LSTM)")
        if os.path.exists(sev_metrics_path):
            with open(sev_metrics_path) as f:
                m = json.load(f)
            k1, k2, k3 = st.columns(3)
            k1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            k2.metric("Macro F1", f"{m['f1_macro']*100:.1f}%")
            k3.metric("Macro Recall", f"{m['recall_macro']*100:.1f}%")
        else:
            st.info("Run `python models/train_severity_model.py` to generate metrics.")

        for img_name, caption in [
            ("severity_accuracy_curve.png", "Training vs Validation Accuracy"),
            ("severity_loss_curve.png", "Training vs Validation Loss"),
            ("severity_confusion_matrix.png", "Confusion Matrix"),
        ]:
            p = os.path.join(MODELS_DIR, img_name)
            if os.path.exists(p):
                st.image(p, caption=caption, use_column_width=True)

# ----------------------------------------------------------------------------
# TAB 3 : THRESHOLD VS DEEP LEARNING
# ----------------------------------------------------------------------------
with tab_compare:
    st.markdown("### ⚖ Traditional Threshold Method vs Deep Learning Method")
    st.caption("Same held-out test windows used for both methods, for a fair comparison.")

    comp_path = os.path.join(MODELS_DIR, "comparison_results.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comp = json.load(f)
        rows = []
        for label, key in [("Traditional Threshold", "threshold_method"), ("Deep Learning (CNN-LSTM)", "deep_learning_method")]:
            d = comp[key]
            rows.append({
                "Method": label,
                "Accuracy": f"{d['accuracy']*100:.1f}%",
                "Precision": f"{d['precision']*100:.1f}%",
                "Recall": f"{d['recall']*100:.1f}%",
                "F1 Score": f"{d['f1_score']*100:.1f}%",
                "False Alarms": d["false_alarms"],
            })
        st.table(pd.DataFrame(rows).set_index("Method"))

        fig = go.Figure()
        metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        thresh_vals = [comp["threshold_method"]["accuracy"], comp["threshold_method"]["precision"],
                       comp["threshold_method"]["recall"], comp["threshold_method"]["f1_score"]]
        dl_vals = [comp["deep_learning_method"]["accuracy"], comp["deep_learning_method"]["precision"],
                   comp["deep_learning_method"]["recall"], comp["deep_learning_method"]["f1_score"]]
        fig.add_trace(go.Bar(name="Threshold Method", x=metrics_names, y=thresh_vals, marker_color="#a4522c"))
        fig.add_trace(go.Bar(name="Deep Learning", x=metrics_names, y=dl_vals, marker_color="#4fd1c5"))
        fig.update_layout(barmode="group", template="plotly_dark", yaxis=dict(range=[0, 1.05]),
                           height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
**Why Deep Learning wins:**
- The threshold rule only looks at a single peak value per window and ignores *how* the sensors evolved over time.
- The CNN-LSTM model learns temporal patterns — e.g. a brief spike that recovers (pothole) vs. a spike followed by sustained abnormal tilt and near-zero speed (real accident) — which sharply reduces false alarms and missed detections.
- The threshold method also needs constant manual re-tuning of magic numbers (3g, 30°, etc.), while the Deep Learning model generalizes from data.
        """)
    else:
        st.info("Run `python models/compare_threshold_vs_dl.py` to generate this comparison.")

# ----------------------------------------------------------------------------
# TAB 4 : ABOUT
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### ℹ️ About This Project

**AI-Based Vehicle Accident Detection, Severity Prediction and Emergency/Service
Center Notification System** is a college-level Deep Learning project built
**entirely as a software simulation** — no physical sensors, ESP32, Raspberry
Pi, Arduino, MQTT, or GSM hardware is used.

**Pipeline:**
1. A rule-based simulator generates realistic multi-sensor time-series data for
   7 driving scenarios (Normal, Hard Braking, Pothole, Sharp Turn, Minor/Major
   Accident, Rollover).
2. A **1D-CNN + LSTM** Deep Learning model analyzes a **50-timestep rolling
   window** of sensor history to detect whether an accident occurred.
3. A second CNN-LSTM model predicts **severity** (Low / Medium / High / Critical).
4. On detection, a simulated GPS location is frozen and the **Haversine
   formula** finds the nearest Hospital, Ambulance, Police Station, Fire
   Station, Vehicle Service Center, and Towing Center from a local JSON
   database.
5. A simulated emergency alert is generated and displayed on this dashboard.

All predictions come from the trained neural networks — **not** from
`if value > threshold` rules. See the "Threshold vs Deep Learning" tab for a
side-by-side comparison proving why this matters.

*Full documentation, architecture diagrams, viva Q&A, limitations and future
scope are provided in the project's README.md.*
    """)
