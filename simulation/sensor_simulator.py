"""
sensor_simulator.py
--------------------
A continuous, stateful sensor stream generator used by the Streamlit dashboard
to simulate a "live" vehicle. It wraps accident_scenarios.py to produce
second-by-second sensor readings and maintains a rolling history buffer
(the "before-accident sensor history") that is fed into the Deep Learning
models.

This module does NOT decide whether an accident occurred - it only produces
realistic raw sensor sequences. Detection/severity is delegated to
prediction/predictor.py (which loads the trained CNN-LSTM models).
"""

from collections import deque
import numpy as np

from simulation.accident_scenarios import generate_event_sequence, EVENT_CLASSES
from simulation.gps_simulator import GPSSimulator

WINDOW_SIZE = 50  # number of timesteps kept as "previous sensor history"

SENSOR_COLUMNS = [
    "speed", "acceleration_x", "acceleration_y", "acceleration_z",
    "gyro_x", "gyro_y", "gyro_z", "vibration", "temperature", "tilt_angle",
]


class SensorSimulator:
    """
    Maintains a sliding window (deque) of the last WINDOW_SIZE sensor readings,
    representing "previous seconds of sensor data" as required by the project
    (BEFORE -> DURING -> AFTER accident analysis).
    """

    def __init__(self, window_size=WINDOW_SIZE, vehicle_id="VH-001"):
        self.window_size = window_size
        self.vehicle_id = vehicle_id
        self.buffer = deque(maxlen=window_size)
        self.gps = GPSSimulator()
        self.history_log = []  # full log for the session (for graphs/timeline)
        self._seed_normal()

    def _seed_normal(self):
        seq = generate_event_sequence("NORMAL", self.window_size)
        for i in range(self.window_size):
            row = {col: seq[col][i] for col in SENSOR_COLUMNS}
            self.buffer.append(row)

    def push_event(self, event_type, n_samples=30):
        """Generate n_samples of a given event type and push them into the
        rolling buffer + full history log. Returns the list of new rows."""
        seq = generate_event_sequence(event_type, n_samples)
        new_rows = []
        is_accident_event = event_type in ("MINOR_ACCIDENT", "MAJOR_ACCIDENT", "ROLLOVER")
        for i in range(n_samples):
            row = {col: float(seq[col][i]) for col in SENSOR_COLUMNS}
            moving = row["speed"] > 2
            lat, lon = self.gps.step(moving=moving)
            row["latitude"] = lat
            row["longitude"] = lon
            row["event_type"] = event_type
            self.buffer.append(row)
            self.history_log.append(row)
            new_rows.append(row)

        if is_accident_event:
            self.gps.freeze()

        return new_rows

    def get_window_array(self):
        """Return the current rolling window as a (window_size, n_features) numpy array."""
        arr = np.array([[row[col] for col in SENSOR_COLUMNS] for row in self.buffer])
        return arr

    def reset(self):
        self.buffer.clear()
        self.history_log = []
        self.gps.reset()
        self._seed_normal()


if __name__ == "__main__":
    sim = SensorSimulator()
    print("Seeded normal window shape:", sim.get_window_array().shape)
    rows = sim.push_event("MAJOR_ACCIDENT", n_samples=20)
    print(f"Generated {len(rows)} MAJOR_ACCIDENT rows. Sample row:")
    print(rows[10])
