"""
compare_threshold_vs_dl.py
----------------------------
Builds the "Traditional Threshold Method vs Deep Learning Method" comparison
required for the college evaluation/presentation.

Traditional method: a simple hard-coded rule -
    IF peak acceleration > 3g OR tilt > 30 deg OR vibration > 6 -> ACCIDENT
This mirrors what a naive embedded system might do without any learning.

Deep Learning method: the trained CNN-LSTM accident_model.keras.

Both are evaluated on the SAME held-out test windows so the comparison is fair.
Results are saved to models/comparison_results.json for the dashboard to display.
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_dataset, build_windows, apply_scaler, load_scaler
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "vehicle_sensor_dataset.csv")
OUT_PATH = os.path.join(BASE_DIR, "comparison_results.json")

SEED = 42


def threshold_predict(X_raw):
    """
    X_raw shape: (n, window, n_features) with FEATURE_COLUMNS order:
    speed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, vibration, temperature, tilt_angle
    """
    acc_x, acc_y, acc_z = X_raw[:, :, 1], X_raw[:, :, 2], X_raw[:, :, 3]
    magnitude_g = np.sqrt(acc_x ** 2 + acc_y ** 2 + (acc_z - 9.8) ** 2) / 9.8
    peak_g = magnitude_g.max(axis=1)
    peak_tilt = X_raw[:, :, 9].max(axis=1)
    peak_vibration = X_raw[:, :, 7].max(axis=1)

    preds = ((peak_g > 3.0) | (peak_tilt > 30.0) | (peak_vibration > 6.0)).astype(int)
    return preds


def main():
    df = load_dataset(DATASET_PATH)
    X, y_acc, _ = build_windows(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_acc, test_size=0.3, random_state=SEED, stratify=y_acc)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    # --- Traditional threshold method ---
    y_pred_thresh = threshold_predict(X_test)
    thresh_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_thresh),
        "precision": precision_score(y_test, y_pred_thresh, zero_division=0),
        "recall": recall_score(y_test, y_pred_thresh, zero_division=0),
        "f1_score": f1_score(y_test, y_pred_thresh, zero_division=0),
        "false_alarms": int(confusion_matrix(y_test, y_pred_thresh)[0, 1]),
    }

    # --- Deep Learning method ---
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, "accident_model.keras"))
    scaler = load_scaler(os.path.join(BASE_DIR, "accident_scaler.joblib"))
    X_test_s = apply_scaler(X_test, scaler)
    y_prob_dl = model.predict(X_test_s, verbose=0).ravel()
    y_pred_dl = (y_prob_dl >= 0.5).astype(int)
    dl_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_dl),
        "precision": precision_score(y_test, y_pred_dl, zero_division=0),
        "recall": recall_score(y_test, y_pred_dl, zero_division=0),
        "f1_score": f1_score(y_test, y_pred_dl, zero_division=0),
        "false_alarms": int(confusion_matrix(y_test, y_pred_dl)[0, 1]),
    }

    results = {"threshold_method": thresh_metrics, "deep_learning_method": dl_metrics}
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved comparison to {OUT_PATH}")


if __name__ == "__main__":
    main()
