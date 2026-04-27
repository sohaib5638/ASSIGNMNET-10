"""
train_cnn_model.py - CNN Model Training Module
Handles all cnn_model architecture, training, and evaluation logic.
Can be run standalone or called from app.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
MODELS_DIR = Path("cnn_models")
DATASET_DIR = Path("dataset")
MODELS_DIR.mkdir(exist_ok=True)

STATUS_PATH = MODELS_DIR / "training_status.json"


# ─── Status Helpers ───────────────────────────────────────────────────────────
def _write_status(state: str, message: str = ""):
    """Write a small status file so the UI can track subprocess lifecycle."""
    with open(STATUS_PATH, "w") as f:
        json.dump({"state": state, "message": message}, f)


# ─── Model Architecture ───────────────────────────────────────────────────────
def build_cnn(num_classes: int, input_shape=(128, 128, 3)) -> keras.Model:
    """Build a custom CNN with batch normalization and dropout."""
    cnn_model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dropout(0.5),
        # Always use softmax + sparse_categorical_crossentropy regardless of class count
        # This avoids the sigmoid/argmax mismatch for binary classification
        layers.Dense(num_classes, activation='softmax'),
    ], name="CustomCNN")

    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return cnn_model


# ─── Data Pipeline ────────────────────────────────────────────────────────────
def get_data_generators(dataset_dir: str, val_split: float = 0.2):
    """Create augmented train/validation generators."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        validation_split=val_split,
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False,
    )

    return train_gen, val_gen


# ─── Live Progress Callback ───────────────────────────────────────────────────
class ProgressCallback(keras.callbacks.Callback):
    """Writes per-epoch metrics to JSON for live dashboard updates."""

    def __init__(self, log_path: str = "cnn_models/training_log.json", total_epochs: int = 20):
        super().__init__()
        self.log_path = log_path
        self.total_epochs = total_epochs
        self.history = {
            "epoch": [], "accuracy": [], "val_accuracy": [],
            "loss": [], "val_loss": [], "total_epochs": total_epochs,
            "complete": False,
        }
        # Write an empty-but-valid log immediately so the UI knows training started
        with open(self.log_path, "w") as f:
            json.dump(self.history, f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["epoch"].append(epoch + 1)
        self.history["accuracy"].append(round(float(logs.get("accuracy", 0)), 6))
        self.history["val_accuracy"].append(round(float(logs.get("val_accuracy", 0)), 6))
        self.history["loss"].append(round(float(logs.get("loss", 0)), 6))
        self.history["val_loss"].append(round(float(logs.get("val_loss", 0)), 6))
        with open(self.log_path, "w") as f:
            json.dump(self.history, f)

    def on_train_end(self, logs=None):
        self.history["complete"] = True
        with open(self.log_path, "w") as f:
            json.dump(self.history, f)


# ─── Training Entry Point ─────────────────────────────────────────────────────
def train(dataset_dir: str = "dataset", epochs: int = 20, val_split: float = 0.2):
    """Full training pipeline."""
    try:
        _write_status("initializing", "Loading dataset and building cnn_model…")

        print(f"\n{'='*50}\n  CNN Training Pipeline\n{'='*50}\n")

        train_gen, val_gen = get_data_generators(dataset_dir, val_split)
        class_names = list(train_gen.class_indices.keys())
        num_classes  = len(class_names)

        print(f"Classes detected : {class_names}")
        print(f"Training samples : {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}\n")

        with open(MODELS_DIR / "class_names.json", "w") as f:
            json.dump(class_names, f)

        cnn_model = build_cnn(num_classes)
        cnn_model.summary()

        progress_cb = ProgressCallback(str(MODELS_DIR / "training_log.json"), epochs)

        callbacks = [
            progress_cb,
            keras.callbacks.ModelCheckpoint(
                str(MODELS_DIR / "best_cnn_model.keras"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7,
                restore_best_weights=True, verbose=1
            ),
        ]

        _write_status("training", "Training in progress…")

        history = cnn_model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Evaluation ────────────────────────────────────────────────────────
        _write_status("evaluating", "Running evaluation…")
        print("\nEvaluating on validation set…")
        val_gen.reset()
        y_true, y_pred_probs = [], []

        for i in range(len(val_gen)):
            x_batch, y_batch = val_gen[i]
            preds = cnn_model.predict(x_batch, verbose=0)
            y_true.extend(y_batch.astype(int).tolist())
            y_pred_probs.extend(preds.tolist())

        y_pred  = np.argmax(y_pred_probs, axis=1)
        y_true  = np.array(y_true)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            linewidths=0.5, linecolor='white',
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(MODELS_DIR / "confusion_matrix.png"), dpi=150, bbox_inches='tight')
        plt.close()

        report    = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        final_acc = float(report['accuracy'])

        eval_results = {
            "final_accuracy": final_acc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "class_names": class_names,
        }

        with open(MODELS_DIR / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        _write_status("complete", f"Training complete. Final accuracy: {final_acc:.4f}")
        print(f"\n✅ Training complete! Final accuracy: {final_acc:.4f}")

        return history.history, class_names, eval_results

    except Exception as exc:
        _write_status("error", str(exc))
        print(f"\n❌ Training error: {exc}")
        raise


# ─── Prediction ───────────────────────────────────────────────────────────────
def load_cnn_model_and_classes():
    """Load trained cnn_model and class names."""
    cnn_model_path   = MODELS_DIR / "best_cnn_model.keras"
    classes_path = MODELS_DIR / "class_names.json"

    if not cnn_model_path.exists():
        raise FileNotFoundError("No trained cnn_model found. Please train first.")

    cnn_model = keras.cnn_models.load_cnn_model(str(cnn_model_path))

    with open(classes_path) as f:
        class_names = json.load(f)

    return cnn_model, class_names


def predict_image(img_array: np.ndarray):
    """
    Predict class of an image array.
    img_array : numpy array (H, W, 3), uint8 0-255
    Returns   : predicted_class (str), confidence (float), all_probs (dict)
    """
    cnn_model, class_names = load_cnn_model_and_classes()

    img = tf.image.resize(img_array, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)

    # cnn_model always outputs softmax probabilities regardless of class count
    probs          = cnn_model.predict(img, verbose=0)[0]
    pred_idx       = int(np.argmax(probs))
    confidence     = float(probs[pred_idx])
    predicted_class = class_names[pred_idx]
    all_probs      = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return predicted_class, confidence, all_probs


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN on custom image dataset")
    parser.add_argument("--dataset",   default="dataset", help="Path to dataset directory")
    parser.add_argument("--epochs",    type=int,   default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.val_split)
