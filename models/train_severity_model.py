"""
train_severity_model.py
-------------------------
Trains a second CNN-LSTM Deep Learning model that takes a sensor window
SURROUNDING A DETECTED EVENT and predicts its severity:
    0 = None, 1 = Low, 2 = Medium, 3 = High, 4 = Critical

The model implicitly learns to weigh:
    - peak acceleration magnitude
    - pre-event speed
    - vibration intensity/duration
    - gyroscope (rotational) change
    - tilt angle
    - how long abnormal motion persists
    - whether the vehicle comes to a stop afterward
purely from the raw time-series, without any hand-written severity rules.
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
                              f1_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils as keras_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import (load_dataset, build_windows, fit_scaler,
                                  apply_scaler, save_scaler, WINDOW_SIZE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "vehicle_sensor_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "severity_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "severity_scaler.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "severity_model_metrics.json")

SEVERITY_NAMES = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
NUM_CLASSES = 5
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_dir):
    hist = history.history
    plt.figure(figsize=(7, 5))
    plt.plot(hist["accuracy"], label="Training Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.title("Severity Model - Accuracy over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "severity_accuracy_curve.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(hist["loss"], label="Training Loss")
    plt.plot(hist["val_loss"], label="Validation Loss")
    plt.title("Severity Model - Loss over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "severity_loss_curve.png"), dpi=140)
    plt.close()


def plot_confusion_matrix(cm, out_dir):
    labels = [SEVERITY_NAMES[i] for i in range(NUM_CLASSES)]
    plt.figure(figsize=(6, 5.5))
    plt.imshow(cm, cmap="Oranges")
    plt.title("Severity Prediction - Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), labels, rotation=45)
    plt.yticks(range(NUM_CLASSES), labels)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "severity_confusion_matrix.png"), dpi=140)
    plt.close()


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print("Building sliding-window sequences...")
    X, _, y_sev = build_windows(df)
    print(f"Total windows: {X.shape[0]}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_sev, test_size=0.3, random_state=SEED, stratify=y_sev)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    scaler = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s = apply_scaler(X_val, scaler)
    X_test_s = apply_scaler(X_test, scaler)
    save_scaler(scaler, SCALER_PATH)

    classes = np.unique(y_train)
    class_weights_arr = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): w for c, w in zip(classes, class_weights_arr)}
    print("Class weights (to handle severity class imbalance):", class_weight)

    model = build_model((X_train_s.shape[1], X_train_s.shape[2]), NUM_CLASSES)
    model.summary()

    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]

    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=20,
        batch_size=128,
        class_weight=class_weight,
        callbacks=cb,
        verbose=2,
    )

    plot_history(history, BASE_DIR)

    print("\nEvaluating on held-out test set...")
    y_prob = model.predict(X_test_s, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    print(f"Accuracy (macro)      : {acc:.4f}")
    print(f"Precision (macro avg) : {prec:.4f}")
    print(f"Recall (macro avg)    : {rec:.4f}")
    print(f"F1 Score (macro avg)  : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    plot_confusion_matrix(cm, BASE_DIR)

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    metrics = {
        "accuracy": acc, "precision_macro": prec, "recall_macro": rec,
        "f1_macro": f1, "confusion_matrix": cm.tolist(),
        "class_names": SEVERITY_NAMES, "n_train": len(X_train),
        "n_val": len(X_val), "n_test": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
