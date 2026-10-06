# 🚨 AI-Based Vehicle Accident Detection, Severity Prediction and Emergency/Service Center Notification System

A **college-level Deep Learning project** that detects vehicle accidents from
time-series sensor data, predicts their severity, and simulates emergency
notification to nearby hospitals, police, and service centers — **entirely
through software simulation**. No physical sensors, ESP32, Raspberry Pi,
Arduino, MQTT, or GSM hardware is required.

---

## 1. Project Story

```
PROBLEM        → Accidents may not be reported immediately
SOLUTION       → AI continuously analyzes vehicle sensor patterns
DETECTION      → Deep Learning detects abnormal accident patterns
SEVERITY       → AI estimates accident severity
LOCATION       → Simulated GPS identifies accident location
RESPONSE       → Nearest emergency facility & service center identified
ALERT          → Automatic (simulated) emergency notification
```

## 2. System Flow

```
BEFORE ACCIDENT
   ↓
Continuous Simulated Sensor Monitoring (speed, accel XYZ, gyro XYZ,
vibration, temperature, tilt, GPS)
   ↓
DURING ACCIDENT
   ↓
CNN-LSTM Deep Learning Accident Detection (on a 50-timestep rolling window)
   ↓
AFTER ACCIDENT
   ↓
CNN-LSTM Severity Prediction (LOW / MEDIUM / HIGH / CRITICAL)
   ↓
Simulated GPS Location Frozen
   ↓
Haversine Distance → Nearest Hospital / Ambulance / Police / Fire / Service / Towing
   ↓
Simulated Emergency Alert + Dashboard Notification
```

## 3. Why CNN-LSTM?

| Layer | Role |
|---|---|
| 1D Convolution | Detects local motifs in the sensor time-series (e.g. "vertical spike + vibration burst" = pothole shape) regardless of exact timestep |
| Max Pooling | Adds noise robustness & mild translation invariance |
| LSTM | Learns how those local patterns *evolve over time* — distinguishing a brief recoverable spike (pothole) from a spike followed by sustained abnormal tilt and near-zero speed (a real accident) |
| Dense + Dropout + Sigmoid/Softmax | Final classification with regularization against overfitting |

A hard-coded `if acceleration > threshold` rule **cannot** capture this
temporal evolution — it can only look at instantaneous or peak values. The
project's **"Threshold vs Deep Learning" dashboard tab** demonstrates this
quantitatively (see Section 9).

## 4. Project Structure

```
vehicle_accident_ai/
│
├── dataset/
│   ├── generate_dataset.py         # synthetic dataset generator
│   └── vehicle_sensor_dataset.csv  # 222,000+ generated sensor records
│
├── simulation/
│   ├── sensor_simulator.py         # live/stateful sensor stream + rolling window
│   ├── gps_simulator.py            # simulated GPS position
│   └── accident_scenarios.py       # rule-based scenario generators (7 event types)
│
├── models/
│   ├── train_accident_model.py     # trains CNN-LSTM accident detector
│   ├── train_severity_model.py     # trains CNN-LSTM severity classifier
│   ├── compare_threshold_vs_dl.py  # threshold vs DL comparison
│   ├── accident_model.keras        # trained model (binary)
│   ├── severity_model.keras        # trained model (5-class)
│   ├── accident_scaler.joblib / severity_scaler.joblib
│   ├── *_metrics.json, comparison_results.json
│   └── *.png                       # accuracy/loss curves, confusion matrices, ROC
│
├── prediction/
│   └── predictor.py                # loads trained models, runs inference
│
├── services/
│   ├── distance_calculator.py      # Haversine formula
│   ├── emergency_service.py        # nearest-facility lookup + alert builder
│   └── service_centers.json        # simulated facility database
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard (main demo UI)
│
├── utils/
│   └── preprocessing.py            # windowing, scaling (shared train/inference)
│
├── requirements.txt
└── README.md
```

## 5. Dataset

`dataset/vehicle_sensor_dataset.csv` — **222,002 synthetically generated
records** (well above the 10,000+ requirement), across 8 simulated vehicles.

> ⚠️ **Academic Disclaimer:** This is 100% synthetically generated sensor
> data created by rule-based generators in `simulation/accident_scenarios.py`,
> for academic simulation only. It is **not** real vehicle telemetry.

**Columns:** `timestamp, vehicle_id, speed, acceleration_x, acceleration_y,
acceleration_z, gyro_x, gyro_y, gyro_z, vibration, temperature, tilt_angle,
latitude, longitude, event_type, accident, severity`

**Event classes:** `NORMAL, HARD_BRAKING, POTHOLE, SHARP_TURN,
MINOR_ACCIDENT, MAJOR_ACCIDENT, ROLLOVER`

**Accident label:** `0` = No Accident, `1` = Accident
**Severity:** `0`=None, `1`=Low, `2`=Medium, `3`=High, `4`=Critical

## 6. Before-Accident Sensor History (Sliding Window)

The model never looks at a single instant. `utils/preprocessing.py` builds
overlapping windows of `WINDOW_SIZE = 50` timesteps (≈ previous 50 seconds at
1 Hz simulated sampling) with a stride of 10. The label of each window is
taken from its *last* timestep, so the model learns to answer: *"given the
last 50 seconds of sensor history, what is happening right now?"* This
directly captures the transition from normal driving → abnormal motion →
collision.

## 7. Model Results (on held-out test set)

**Accident Detection (CNN-LSTM, binary):**
Accuracy ≈ 100%, Recall ≈ 100%, Precision ≈ 100%, ROC-AUC ≈ 1.00, **0 false
negatives** on the test split.

**Severity Prediction (CNN-LSTM, 5-class):**
Accuracy ≈ 99.7%, Macro F1 ≈ 99.4%.

> 📌 **Academic honesty note:** These near-perfect scores are expected
> because the synthetic scenarios are generated with clearly separable
> statistical signatures (by design, for teaching clarity). On **real-world
> noisy sensor data**, expect meaningfully lower — but still strong —
> performance, and recall/false-negative analysis becomes even more
> important. This is explicitly discussed in the "Project Limitations"
> section below and is good material for your viva.

Full accuracy/loss curves, confusion matrices, and ROC curve are saved as
`.png` files in `models/` and are shown live in the dashboard's
"Model Performance" tab.

## 8. Real-Time Prediction Pipeline

`prediction/predictor.py` is the **only** place a final decision is made —
by the trained neural networks, never by a hard-coded rule:

```
Sensor Sequence → Preprocessing (scaling) → CNN-LSTM → Prediction Probability → Accident Classification
```

## 9. Threshold vs Deep Learning Comparison

`models/compare_threshold_vs_dl.py` evaluates a naive rule
(`peak_accel > 3g OR tilt > 30° OR vibration > 6`) against the trained
CNN-LSTM on the **same** test windows:

| Method | Accuracy | Precision | Recall | F1 | False Alarms |
|---|---|---|---|---|---|
| Traditional Threshold | ~87% | ~88% | ~74% | ~80% | 120 |
| Deep Learning (CNN-LSTM) | ~100% | ~100% | ~100% | ~100% | 0 |

The threshold method misses more real accidents (lower recall) and raises
more false alarms, because it only checks a single peak value and ignores
temporal context.

## 10. Installation

```bash
cd vehicle_accident_ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 11. Execution Instructions

Run these once, in order, to (re)build the dataset and models (already
included in this deliverable, but reproducible from scratch):

```bash
# 1. Generate the synthetic dataset
python dataset/generate_dataset.py

# 2. Train the accident detection model
python models/train_accident_model.py

# 3. Train the severity prediction model
python models/train_severity_model.py

# 4. Generate the threshold-vs-DL comparison
python models/compare_threshold_vs_dl.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## 12. Using the Dashboard

1. Use the **sidebar** to trigger a scenario: Normal Driving, Hard Braking,
   Pothole, Sharp Turn, Minor Accident, Major Accident, or Rollover.
2. Watch the **live sensor graphs**, **AI Accident Detection** card, and
   **Severity Prediction** card update in real time.
3. On accident detection: simulated GPS location freezes, nearest Hospital /
   Ambulance / Police / Fire / Service / Towing centers are computed via the
   Haversine formula, and a simulated emergency alert is generated.
4. Try the **Driver Response Simulation** ("I'm Safe" / "Send Emergency Alert"
   / "Simulate No Response").
5. Explore **Model Performance** and **Threshold vs Deep Learning** tabs for
   the evaluation story.

## 13. Sample Output

```
🚨 ACCIDENT ALERT (SIMULATED)
Vehicle ID: VH-001
Accident: DETECTED   Confidence: 96.8%
Severity: HIGH        Confidence: 94.7%
Speed Before Impact: 78 km/h
Peak Acceleration: 6.4 g
Tilt: 43°
GPS: 28.6139, 77.2090
Vehicle Status: STOPPED

Notification Sent To:
✓ Emergency Contact   ✓ Nearest Hospital
✓ Nearest Police Station   ✓ Vehicle Service Center

Disclaimer: This is a simulated notification for academic
demonstration only. No real emergency services were contacted.
```

## 14. Use-Case Diagram (textual)

```
                        ┌────────────────────────┐
                        │        Driver           │
                        └───────────┬─────────────┘
                                    │
                     ┌──────────────┼───────────────┐
                     │              │               │
             (views dashboard)  (responds to    (triggers demo
                     │           safety check)    scenarios)
                     ▼              ▼               ▼
             ┌───────────────────────────────────────────┐
             │        AI Vehicle Accident System          │
             │  - Sensor Simulation                       │
             │  - CNN-LSTM Accident Detection              │
             │  - CNN-LSTM Severity Prediction             │
             │  - GPS Simulation                           │
             │  - Nearest Facility Lookup (Haversine)      │
             │  - Simulated Emergency Alert Generation     │
             └───────────────┬─────────────────────────────┘
                              │ (simulated notification)
                              ▼
             ┌───────────────────────────────────────────┐
             │  Hospital / Police / Fire / Service Center  │
             │        (simulated recipients)               │
             └───────────────────────────────────────────┘
```

## 15. Project Limitations

- All sensor data is synthetically generated; real accelerometer/gyroscope
  data from actual vehicles would contain more noise, sensor drift, and
  ambiguous edge cases.
- The system assumes a fixed 1 Hz sampling rate and a single vehicle type.
- GPS simulation does not model real road networks or map-matching.
- Emergency notifications are simulated only — no real SMS/GSM/call
  integration exists in this project.
- The models were trained on 7 discrete event archetypes; real-world
  accidents lie on a continuum and may not match these exact patterns.

## 16. Future Scope

- Integrate with real IMU/accelerometer data from smartphones for
  pilot testing.
- Add a Transformer-based (attention) model to compare against CNN-LSTM.
- Real-time streaming ingestion (e.g., Kafka) for multi-vehicle fleets.
- Integration with real mapping APIs and live traffic data.
- On-device (edge) deployment via TensorFlow Lite for low-latency detection.
- Multi-modal fusion with dash-cam video for visual accident confirmation.

## 17. Viva Questions & Answers

**Q1. Why use CNN-LSTM instead of a simple threshold rule?**
A threshold only checks instantaneous/peak values and cannot distinguish a
brief recoverable event (e.g., pothole) from a genuine collision that leaves
the vehicle stopped and tilted. CNN-LSTM learns the temporal *shape* and
*evolution* of the sensor signals, which is far more robust.

**Q2. Why 1D CNN before the LSTM instead of feeding raw signals directly to the LSTM?**
The 1D convolution layers act as automatic local feature extractors — they
learn short motifs (like a spike-then-decay shape) without needing hand
engineering, and reduce the sequence length/computation before the LSTM
models longer-term temporal dependencies.

**Q3. Why is recall emphasized over accuracy for accident detection?**
Because this is a safety-critical system: a missed accident (false negative)
could delay life-saving help, which is far more costly than an occasional
false alarm. High recall minimizes missed accidents.

**Q4. What is the sliding window and why WINDOW_SIZE = 50?**
It's the number of previous timesteps (≈ 50 seconds at 1 Hz) fed into the
model per prediction, so the model can see the full before → during
transition, not just a single reading. The size can be tuned based on
sampling rate and how much history is meaningfully predictive.

**Q5. How is severity determined, and why not use rules like "tilt > 40° = Critical"?**
The severity model learns a *combination* of peak acceleration, pre-crash
speed, vibration, gyroscope change, tilt, and how long abnormal motion
persists — jointly, from data — rather than relying on any single hard
threshold, so it captures interactions no single rule could.

**Q6. How is the accident location determined without real GPS hardware?**
`gps_simulator.py` produces continuously updating simulated coordinates,
and the moment an accident is detected, the current position is "frozen" and
used as the official accident location — mirroring how a real GPS module
would behave.

**Q7. How are nearby facilities found?**
Via the Haversine formula (`services/distance_calculator.py`), which computes
great-circle distance between the accident's GPS coordinates and each
facility in the simulated `service_centers.json` database, without needing
any real mapping API.

**Q8. What would need to change to deploy this on a real vehicle?**
Replace the simulated sensor/GPS modules with real IMU/GPS hardware readers,
retrain/fine-tune on real (noisy) sensor data, add robust error handling and
sensor fault detection, and connect the notification module to real
SMS/GSM/telephony APIs.

**Q9. Why Streamlit instead of a full web stack (React/Django)?**
This is a Deep Learning-focused project, not a CRUD/web-development project.
Streamlit lets us build an interactive, professional dashboard in pure
Python with minimal boilerplate, keeping the focus on the ML pipeline.

**Q10. How do you know the model isn't just memorizing simple thresholds internally?**
The comparison in `models/compare_threshold_vs_dl.py` shows the CNN-LSTM
achieves higher recall/precision with fewer false alarms than an explicit
threshold rule evaluated on the same data — if the network had just learned
an equivalent threshold, both methods would perform similarly.

---

*This project is intended purely as a college Deep Learning
demonstration/prototype. It is not a certified or commercial automotive
safety product.*
