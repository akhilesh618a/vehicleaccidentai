"""
preprocessing.py
-----------------
Shared preprocessing utilities for turning the flat sensor CSV into
overlapping time-series windows suitable for the CNN-LSTM models, plus
feature scaling helpers that are reused identically at training and
inference time (so the live dashboard predictions match training conditions).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "speed", "acceleration_x", "acceleration_y", "acceleration_z",
    "gyro_x", "gyro_y", "gyro_z", "vibration", "temperature", "tilt_angle",
]

WINDOW_SIZE = 50   # ~50 timesteps of "previous sensor history" per the project spec
STEP_SIZE = 10     # stride between consecutive windows (controls overlap / dataset size)


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def build_windows(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                   group_col="vehicle_id"):
    """
    Slice the continuous per-vehicle sensor stream into overlapping windows.
    Each window's label (accident / severity) is taken from the LAST timestep
    in the window, i.e. "given the last `window_size` seconds of sensor
    history, what's happening right now?" - this directly implements the
    project's "before-accident sensor history -> prediction" requirement.
    """
    X, y_accident, y_severity = [], [], []

    for _, group in df.groupby(group_col, sort=False):
        group = group.reset_index(drop=True)
        feats = group[FEATURE_COLUMNS].values
        acc_labels = group["accident"].values
        sev_labels = group["severity"].values

        n = len(group)
        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            X.append(feats[start:end])
            y_accident.append(acc_labels[end - 1])
            y_severity.append(sev_labels[end - 1])

    return np.array(X), np.array(y_accident), np.array(y_severity)


def fit_scaler(X_train):
    """Fit a StandardScaler on flattened training windows."""
    n_samples, n_steps, n_features = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    return scaler


def apply_scaler(X, scaler):
    n_samples, n_steps, n_features = X.shape
    X_scaled = scaler.transform(X.reshape(-1, n_features))
    return X_scaled.reshape(n_samples, n_steps, n_features)


def save_scaler(scaler, path):
    joblib.dump(scaler, path)


def load_scaler(path):
    return joblib.load(path)
