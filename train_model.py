"""
train_model.py
---------------
TASK 2: Emotion Recognition from Speech (model training)

Loads the MFCC features from data/features.npz and trains a hybrid
CNN + LSTM deep learning model:
  - Conv1D layers learn local spectral patterns across MFCC coefficients
  - LSTM layer learns the temporal dynamics of speech across frames
  - Dense + softmax head classifies the emotion

Run:
    python extract_features.py   # only needed once
    python train_model.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

RANDOM_STATE = 42
FEATURES_PATH = os.path.join("data", "features.npz")
OUT_DIR = "outputs"

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"'{FEATURES_PATH}' not found. Run `python extract_features.py` first."
        )
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X, y = data["X"], data["y"]
    label_map = data["label_map"].item()
    inv_label_map = {v: k for k, v in label_map.items()}
    return X, y, inv_label_map


def build_model(input_shape, n_classes):
    """CNN (spatial/spectral patterns) + LSTM (temporal dynamics) hybrid."""
    inputs = layers.Input(shape=input_shape)  # (n_coeffs, time, 1)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Reshape (freq, time, channels) -> (time, features) for the LSTM
    shape = x.shape
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((shape[2], shape[1] * shape[3]))(x)

    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y, inv_label_map = load_features()
    n_classes = len(inv_label_map)
    class_names = [inv_label_map[i] for i in range(n_classes)]

    # Normalize features (per-sample standardization across all MFCC values)
    scaler = StandardScaler()
    n_samples, n_coeffs, n_time = X.shape
    X_flat = X.reshape(n_samples, -1)
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(n_samples, n_coeffs, n_time, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model(input_shape=X.shape[1:], n_classes=n_classes)
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=16,
        callbacks=[early_stop],
        verbose=2,
    )

    # --- Evaluation ---
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Emotion Recognition")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    model.save(os.path.join(OUT_DIR, "emotion_recognition_model.keras"))
    import joblib
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
        json.dump(inv_label_map, f, indent=2)

    print(f"\nModel, scaler, label map, plots and metrics saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
