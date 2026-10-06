"""
generate_dataset.py
--------------------
Generates a large synthetic time-series vehicle sensor dataset for academic
simulation purposes. This is NOT real vehicle data - every row is produced
by the rule-based scenario generator in simulation/accident_scenarios.py.

Output: dataset/vehicle_sensor_dataset.csv
Columns:
    timestamp, vehicle_id, speed, acceleration_x, acceleration_y, acceleration_z,
    gyro_x, gyro_y, gyro_z, vibration, temperature, tilt_angle,
    latitude, longitude, event_type, accident, severity
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.accident_scenarios import generate_event_sequence, EVENT_CLASSES
from simulation.gps_simulator import GPSSimulator

np.random.seed(42)

# How many "trips" (episodes) to simulate per event type, and how long each is.
EVENT_EPISODE_LENGTHS = {
    "NORMAL": (200, 400),          # majority-class baseline driving
    "HARD_BRAKING": (40, 70),
    "POTHOLE": (20, 35),
    "SHARP_TURN": (40, 70),
    "MINOR_ACCIDENT": (60, 90),
    "MAJOR_ACCIDENT": (60, 90),
    "ROLLOVER": (60, 90),
}

EVENT_EPISODE_COUNTS = {
    "NORMAL": 35,
    "HARD_BRAKING": 55,
    "POTHOLE": 55,
    "SHARP_TURN": 55,
    "MINOR_ACCIDENT": 45,
    "MAJOR_ACCIDENT": 45,
    "ROLLOVER": 40,
}

N_VEHICLES = 8
VEHICLE_IDS = [f"VH-{i:03d}" for i in range(1, N_VEHICLES + 1)]


def build_dataset():
    rows = []
    start_time = datetime(2026, 1, 1, 8, 0, 0)

    for vehicle_id in VEHICLE_IDS:
        gps = GPSSimulator(
            base_lat=28.6139 + np.random.uniform(-0.3, 0.3),
            base_lon=77.2090 + np.random.uniform(-0.3, 0.3),
        )
        current_time = start_time + timedelta(minutes=np.random.randint(0, 500))

        for event_type in EVENT_CLASSES:
            n_episodes = EVENT_EPISODE_COUNTS[event_type]
            lo, hi = EVENT_EPISODE_LENGTHS[event_type]

            for _ in range(n_episodes):
                n_samples = np.random.randint(lo, hi)
                seq = generate_event_sequence(event_type, n_samples)
                is_accident_event = event_type in ("MINOR_ACCIDENT", "MAJOR_ACCIDENT", "ROLLOVER")

                for i in range(n_samples):
                    moving = seq["speed"][i] > 2
                    lat, lon = gps.step(moving=moving)
                    row = {
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "vehicle_id": vehicle_id,
                        "speed": round(float(seq["speed"][i]), 2),
                        "acceleration_x": round(float(seq["acceleration_x"][i]), 4),
                        "acceleration_y": round(float(seq["acceleration_y"][i]), 4),
                        "acceleration_z": round(float(seq["acceleration_z"][i]), 4),
                        "gyro_x": round(float(seq["gyro_x"][i]), 3),
                        "gyro_y": round(float(seq["gyro_y"][i]), 3),
                        "gyro_z": round(float(seq["gyro_z"][i]), 3),
                        "vibration": round(float(seq["vibration"][i]), 3),
                        "temperature": round(float(seq["temperature"][i]), 2),
                        "tilt_angle": round(float(seq["tilt_angle"][i]), 2),
                        "latitude": round(lat, 6),
                        "longitude": round(lon, 6),
                        "event_type": event_type,
                        "accident": seq["accident"],
                        "severity": seq["severity"],
                    }
                    rows.append(row)
                    current_time += timedelta(seconds=1)

                if is_accident_event:
                    gps.reset()  # start a fresh "trip" location after an accident episode

    df = pd.DataFrame(rows)
    return df


def main():
    df = build_dataset()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "vehicle_sensor_dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"Dataset generated: {out_path}")
    print(f"Total records: {len(df)}")
    print("\nEvent type distribution:")
    print(df["event_type"].value_counts())
    print("\nAccident label distribution:")
    print(df["accident"].value_counts())
    print("\nSeverity label distribution:")
    print(df["severity"].value_counts().sort_index())


if __name__ == "__main__":
    main()
