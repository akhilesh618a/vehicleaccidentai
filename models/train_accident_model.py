"""
train_accident_model.py
------------------------
Trains a 1D-CNN + LSTM Deep Learning model to classify whether a window of
sensor readings (the previous ~50 timesteps) represents an ACCIDENT or NOT.

WHY CNN-LSTM FOR THIS DATA?
  - The 1D convolution layers act as local pattern detectors across the
    sensor time-series: they learn short motifs such as "sudden vertical
    spike + vibration burst" (a pothole) or "sharp deceleration + gyro
    spike" (a collision), regardless of exactly which timestep they occur at.
  - Max pooling condenses these motifs and adds translation invariance and
    noise robustness.
  - The LSTM layer then models how these local patterns evolve over time -
    e.g. distinguishing "brief spike that recovers" (pothole) from
    "spike followed by sustained abnormal tilt and near-zero speed"
    (a real accident). This temporal memory is exactly what a hard-coded
    if/threshold rule cannot capture.
  - Together, CNN-LSTM learns BOTH the shape of an event and its evolution
    through time, which is why it generalizes far better than single-value
    thresholds on acceleration or vibration.

Because accident detection is safety-critical, this script explicitly reports
RECALL and false-negative analysis in addition to accuracy.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import (load_dataset, build_windows, fit_scaler,
                                  apply_scaler, save_scaler, WINDOW_SIZE, FEATURE_COLUMNS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "vehicle_sensor_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "accident_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "accident_scaler.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "accident_model_metrics.json")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        # metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
        #          tf.keras.metrics.Precision(name="precision")],
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_dir):
    hist = history.history

    plt.figure(figsize=(7, 5))
    plt.plot(hist["accuracy"], label="Training Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.title("Accident Model - Accuracy over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accident_accuracy_curve.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(hist["loss"], label="Training Loss")
    plt.plot(hist["val_loss"], label="Validation Loss")
    plt.title("Accident Model - Loss over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accident_loss_curve.png"), dpi=140)
    plt.close()


def plot_confusion_matrix(cm, out_dir):
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Accident Detection - Confusion Matrix")
    plt.colorbar()
    labels = ["No Accident", "Accident"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accident_confusion_matrix.png"), dpi=140)
    plt.close()


def plot_roc(y_true, y_prob, auc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Accident Detection - ROC Curve")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accident_roc_curve.png"), dpi=140)
    plt.close()


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print("Building sliding-window sequences (WINDOW_SIZE={})...".format(WINDOW_SIZE))
    X, y_acc, _ = build_windows(df)
    print(f"Total windows: {X.shape[0]}  | Shape per window: {X.shape[1:]}")
    print(f"Accident windows: {int(y_acc.sum())}  | Normal windows: {int(len(y_acc) - y_acc.sum())}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_acc, test_size=0.3, random_state=SEED, stratify=y_acc)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    scaler = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s = apply_scaler(X_val, scaler)
    X_test_s = apply_scaler(X_test, scaler)
    save_scaler(scaler, SCALER_PATH)

    model = build_model((X_train_s.shape[1], X_train_s.shape[2]))
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=20,
        batch_size=128,
        callbacks=cb,
        verbose=2,
    )

    plot_history(history, BASE_DIR)

    print("\nEvaluating on held-out test set...")
    y_prob = model.predict(X_test_s, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}  <-- most safety-critical metric")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"False Negatives (missed accidents): {fn}  "
          f"(False Negative Rate: {false_negative_rate:.4f})")

    plot_confusion_matrix(cm, BASE_DIR)
    plot_roc(y_test, y_prob, auc, BASE_DIR)

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "roc_auc": auc, "confusion_matrix": cm.tolist(),
        "false_negatives": int(fn), "false_negative_rate": false_negative_rate,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
