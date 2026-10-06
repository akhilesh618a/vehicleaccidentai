"""
predictor.py
-------------
Loads the trained CNN-LSTM accident-detection and severity-prediction models
(plus their fitted scalers) and exposes a simple API for running inference
on a live sensor window. This is the ONLY place in the whole pipeline where
the final ACCIDENT / SEVERITY decision is made - and it is made entirely
by the trained Deep Learning models, never by a hard-coded threshold.
"""

import os
import numpy as np
import tensorflow as tf

from utils.preprocessing import apply_scaler, load_scaler, FEATURE_COLUMNS, WINDOW_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

ACCIDENT_MODEL_PATH = os.path.join(MODELS_DIR, "accident_model.keras")
ACCIDENT_SCALER_PATH = os.path.join(MODELS_DIR, "accident_scaler.joblib")
SEVERITY_MODEL_PATH = os.path.join(MODELS_DIR, "severity_model.keras")
SEVERITY_SCALER_PATH = os.path.join(MODELS_DIR, "severity_scaler.joblib")

SEVERITY_NAMES = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}


class AccidentPredictor:
    """Wraps both trained models for easy reuse by the Streamlit dashboard
    or any other client script."""

    def __init__(self):
        self.accident_model = tf.keras.models.load_model(ACCIDENT_MODEL_PATH, compile=False)
        self.accident_scaler = load_scaler(ACCIDENT_SCALER_PATH)
        self.severity_model = tf.keras.models.load_model(SEVERITY_MODEL_PATH, compile=False)
        self.severity_scaler = load_scaler(SEVERITY_SCALER_PATH)

    def _prep(self, window, scaler):
        """window: numpy array of shape (WINDOW_SIZE, n_features)"""
        window = np.asarray(window, dtype=np.float32)
        if window.shape[0] < WINDOW_SIZE:
            pad = np.repeat(window[:1], WINDOW_SIZE - window.shape[0], axis=0)
            window = np.vstack([pad, window])
        elif window.shape[0] > WINDOW_SIZE:
            window = window[-WINDOW_SIZE:]
        window = window[np.newaxis, ...]  # (1, WINDOW_SIZE, n_features)
        return apply_scaler(window, scaler)

    def predict_accident(self, window):
        """Returns (is_accident: bool, probability: float)"""
        X = self._prep(window, self.accident_scaler)
        prob = float(self.accident_model.predict(X, verbose=0).ravel()[0])
        return prob >= 0.5, prob

    def predict_severity(self, window):
        """Returns (severity_label: str, confidence: float, class_probs: np.array)"""
        X = self._prep(window, self.severity_scaler)
        probs = self.severity_model.predict(X, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        return SEVERITY_NAMES[class_idx], float(probs[class_idx]), probs

    def full_prediction(self, window):
        """Run both models and return a combined result dict."""
        is_accident, acc_prob = self.predict_accident(window)
        result = {
            "accident_detected": is_accident,
            "accident_probability": acc_prob,
            "severity": "NONE",
            "severity_confidence": 1.0 - acc_prob,
        }
        if is_accident:
            sev_label, sev_conf, _ = self.predict_severity(window)
            result["severity"] = sev_label
            result["severity_confidence"] = sev_conf
        return result


_predictor_singleton = None


def get_predictor():
    """Lazy singleton so Streamlit doesn't reload the models on every rerun."""
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = AccidentPredictor()
    return _predictor_singleton
