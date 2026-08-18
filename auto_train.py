#!/usr/bin/env python3
"""HeartLens AI — Automated Training & PDF Report Pipeline.

Orchestrates end-to-end model training (denoiser + classifier),
evaluates both models, generates visualisations, and produces a
self-contained PDF report with all results.

Usage:
    python3 auto_train.py --epochs 50 --max-per-class 3000 [--data-dir ./mitdb]
    python3 auto_train.py --gui
"""

# ═══════════════════════════════════════════════════════════════════
#  BOOTSTRAP: auto-install missing dependencies
# ═══════════════════════════════════════════════════════════════════

_REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "tensorflow": "tensorflow",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "fpdf": "fpdf2",
    "scipy": "scipy",
    "wfdb": "wfdb",
}

_installed_any = False
for import_name, pip_name in _REQUIRED_PACKAGES.items():
    try:
        __import__(import_name)
    except ImportError:
        import subprocess, sys
        print(f"  Installing missing dependency: {pip_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"])
        _installed_any = True

if _installed_any:
    print("  All dependencies installed. Restarting...\n")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ═══════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════

import argparse, datetime, json, os, sys, textwrap, io, traceback, threading, time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

# ── matplotlib must be configured before any pyplot import ──────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── fpdf2 for PDF report generation ────────────────────────────────
from fpdf import FPDF

# ── tkinter for GUI mode ───────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    _HAVE_TK = True
except ImportError:
    _HAVE_TK = False

# ── Local modules ───────────────────────────────────────────────────
TRAIN_DIR = Path(__file__).parent / "heart-lens-training"
sys.path.insert(0, str(TRAIN_DIR))

from data_loader import (load_all_segments, load_record_segments,
                          split_dataset, split_by_patient,
                          WINDOW_SAMPLES, WINDOW_SEC, NUM_CLASSES,
                          RECORDS_LIST)
from noise_pipeline import add_all_noise

# ── Reproducibility ─────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Constants ───────────────────────────────────────────────────────
CLASS_NAMES = ["Normal", "APB", "PVC"]
# Class 1 is atrial premature beat (APB), honestly labeled. True AF
# rhythm detection uses afdb_loader.py (MIT-BIH Atrial Fibrillation DB).
NOISE_LEVELS = (0, 5, 10, 15, 20, 30, 40)
VAL_NOISE_SNR = 15
SEQ_LEN = WINDOW_SAMPLES  # 360 (1 s at 360 Hz)

OUT_DIR = Path(__file__).parent / "auto_train_output"
FIGS_DIR = OUT_DIR / "figures"
os.makedirs(FIGS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_denoiser(input_shape=(SEQ_LEN, 1)):
    """Conv1D encoder-decoder denoiser."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="denoiser_input")
    x = tf.keras.layers.Conv1D(16, 15, padding="same",
                                activation="relu", name="enc_conv1")(inputs)
    x = tf.keras.layers.MaxPool1D(2, name="enc_pool1")(x)
    x = tf.keras.layers.Conv1D(8, 15, padding="same",
                                activation="relu", name="enc_conv2")(x)
    x = tf.keras.layers.MaxPool1D(2, name="enc_pool2")(x)
    x = tf.keras.layers.UpSampling1D(2, name="dec_upsample1")(x)
    x = tf.keras.layers.Conv1D(8, 15, padding="same",
                                activation="relu", name="dec_conv1")(x)
    x = tf.keras.layers.UpSampling1D(2, name="dec_upsample2")(x)
    x = tf.keras.layers.Conv1D(1, 15, padding="same", name="dec_conv2")(x)
    model = tf.keras.Model(inputs, x, name="conv1d_denoiser")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_classifier(input_shape=(SEQ_LEN, 1), num_classes=NUM_CLASSES):
    """1D-CNN with 3 Conv blocks -> GAP -> Dense -> Softmax."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="classifier_input")
    x = tf.keras.layers.Conv1D(32, 5, padding="same",
                                activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPool1D(2, name="pool1")(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same",
                                activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPool1D(2, name="pool2")(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same",
                                activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                     name="classifier_output")(x)
    model = tf.keras.Model(inputs, outputs, name="cnn_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ═══════════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def train_denoiser(data_dir, max_per_class, epochs):
    print("\n" + "=" * 60)
    print("DENOISER TRAINING")
    print("=" * 60)

    by_class = load_all_segments(data_dir, max_per_class=max_per_class)
    all_segs = []
    for segs in by_class.values():
        all_segs.extend(segs)
    all_segs = np.array(all_segs).reshape(-1, SEQ_LEN, 1)
    print(f"  Total segments loaded: {all_segs.shape[0]}")

    X_train, X_tmp = train_test_split(all_segs, test_size=0.3,
                                       random_state=42)
    X_val, X_test = train_test_split(X_tmp, test_size=0.5,
                                      random_state=42)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, "
          f"Test: {X_test.shape}")

    print("  Generating noisy augmentations...")
    X_clean_aug, X_noisy_aug = [], []
    for ecg in X_train:
        for snr in NOISE_LEVELS:
            noisy = add_all_noise(ecg.flatten(), snr_db=snr)
            X_clean_aug.append(ecg.flatten())
            X_noisy_aug.append(noisy)
    X_clean_aug = np.array(X_clean_aug).reshape(-1, SEQ_LEN, 1)
    X_noisy_aug = np.array(X_noisy_aug).reshape(-1, SEQ_LEN, 1)
    print(f"  Augmented: {X_noisy_aug.shape}")

    X_val_noisy = np.array([add_all_noise(e.flatten(), snr_db=VAL_NOISE_SNR)
                            for e in X_val]).reshape(-1, SEQ_LEN, 1)

    model = build_denoiser()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5,
                                          restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]
    history = model.fit(
        X_noisy_aug, X_clean_aug,
        validation_data=(X_val_noisy, X_val),
        epochs=epochs, batch_size=64,
        callbacks=callbacks, verbose=2
    )

    # Evaluate
    X_test_noisy = np.array([add_all_noise(e.flatten(), snr_db=VAL_NOISE_SNR)
                             for e in X_test]).reshape(-1, SEQ_LEN, 1)
    test_loss, test_mae = model.evaluate(X_test_noisy, X_test, verbose=0)

    # Convert to int8 TFLite
    print("  Converting to int8 TFLite...")
    def rep_data():
        for ecg in X_val[:100]:
            noisy = add_all_noise(ecg.flatten(), snr_db=VAL_NOISE_SNR)
            yield [noisy.reshape(1, SEQ_LEN, 1).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    return {
        "model": model,
        "tflite": tflite_model,
        "tflite_size_kb": len(tflite_model) / 1024,
        "history": history,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "X_test": X_test,
        "X_test_noisy": X_test_noisy,
    }


def train_classifier(data_dir, max_per_class, epochs):
    print("\n" + "=" * 60)
    print("CLASSIFIER TRAINING")
    print("=" * 60)

    by_class = load_all_segments(data_dir, max_per_class=max_per_class)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        split_dataset(by_class)

    print(f"  Train: {X_train.shape}, y dist: {np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}, y dist: {np.bincount(y_val)}")
    print(f"  Test:  {X_test.shape}, y dist: {np.bincount(y_test)}")

    # Class weights
    classes = np.unique(y_train)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"  Class weights: {class_weight_dict}")

    model = build_classifier()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10,
                                          restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks, verbose=2
    )

    # Float32 evaluation
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    labels_present = sorted(set(y_test) | set(y_pred))
    report_dict = classification_report(y_test, y_pred,
                                         labels=list(range(NUM_CLASSES)),
                                         target_names=CLASS_NAMES,
                                         output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred,
                          labels=list(range(NUM_CLASSES)))
    f1_float = f1_score(y_test, y_pred, average=None,
                         labels=list(range(NUM_CLASSES)),
                         zero_division=0)

    # Convert to int8 TFLite
    print("  Converting to int8 TFLite...")
    def rep_data():
        for _ in range(200):
            idx = np.random.randint(0, len(X_val))
            yield [X_val[idx:idx + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Quantized validation
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    in_scale, in_zero = in_d["quantization"]
    out_scale, out_zero = out_d["quantization"]

    y_pred_q = []
    for i in range(len(X_test)):
        x_float = X_test[i].astype(np.float32)
        x_q = np.round(x_float / in_scale + in_zero).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(in_d["index"],
                               x_q.reshape(1, SEQ_LEN, 1))
        interpreter.invoke()
        raw = interpreter.get_tensor(out_d["index"])[0]
        out = (raw.astype(np.float32) - out_zero) * out_scale
        y_pred_q.append(np.argmax(out))
    y_pred_q = np.array(y_pred_q)

    f1_quant = f1_score(y_test, y_pred_q, average=None,
                         labels=list(range(NUM_CLASSES)),
                         zero_division=0)
    report_dict_q = classification_report(y_test, y_pred_q,
                                           labels=list(range(NUM_CLASSES)),
                                           target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0)
    cm_q = confusion_matrix(y_test, y_pred_q,
                            labels=list(range(NUM_CLASSES)))

    return {
        "model": model,
        "tflite": tflite_model,
        "tflite_size_kb": len(tflite_model) / 1024,
        "history": history,
        "report_dict": report_dict,
        "report_dict_q": report_dict_q,
        "cm": cm,
        "cm_q": cm_q,
        "f1_float": f1_float,
        "f1_quant": f1_quant,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_q": y_pred_q,
        "in_quant": (in_scale, in_zero),
        "out_quant": (out_scale, out_zero),
    }


# ═══════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════

def plot_training_history(history, title, save_path):
    metrics = history.history.keys()
    loss_key = "loss"
    secondary = [m for m in metrics if m != loss_key and not m.startswith("val_")]
    secondary = secondary[:1]  # pick first non-loss metric
    plot_metrics = [loss_key]

    ncols = len(plot_metrics) + len(secondary)
    fig, axes = plt.subplots(1, max(ncols, 1), figsize=(5 * max(ncols, 1), 4))
    if ncols == 1:
        axes = [axes]

    for ax, metric in zip(axes, plot_metrics + secondary):
        val_key = f"val_{metric}"
        train_data = history.history.get(metric, [])
        val_data = history.history.get(val_key, [])
        if train_data:
            ax.plot(train_data, label=f"train_{metric}")
        if val_data:
            ax.plot(val_data, label=f"val_{metric}")
        ax.set_title(f"{title} — {metric.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        if train_data or val_data:
            ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=title, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_denoiser_samples(X_test, X_test_noisy, model, save_path, n=4):
    denoised = model.predict(X_test_noisy[:n], verbose=0)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.5 * n))

    for i in range(n):
        t = np.arange(SEQ_LEN) / 360.0
        axes[i, 0].plot(t, X_test[i, :, 0], color="green", alpha=0.8)
        axes[i, 0].set_ylabel(f"Sample {i+1}")
        if i == 0: axes[i, 0].set_title("Clean (target)")
        axes[i, 1].plot(t, X_test_noisy[i, :, 0], color="red", alpha=0.6)
        if i == 0: axes[i, 1].set_title(f"Noisy (SNR={VAL_NOISE_SNR} dB)")
        axes[i, 2].plot(t, denoised[i, :, 0], color="blue", alpha=0.8)
        if i == 0: axes[i, 2].set_title("Denoised")
        if i == n - 1:
            axes[i, 0].set_xlabel("Time (s)")
            axes[i, 1].set_xlabel("Time (s)")
            axes[i, 2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_class_distribution(y_train, y_val, y_test, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, np.bincount(y_train, minlength=len(class_names)),
           width, label="Train", alpha=0.8)
    ax.bar(x, np.bincount(y_val, minlength=len(class_names)),
           width, label="Validation", alpha=0.8)
    ax.bar(x + width, np.bincount(y_test, minlength=len(class_names)),
           width, label="Test", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Segments")
    ax.set_title("Class Distribution per Split")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  PDF REPORT
# ═══════════════════════════════════════════════════════════════════

class HeartLensReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 6, "HeartLens AI - Automated Training Report", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body_text(self, txt):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, txt)
        self.ln(2)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.cell(70, 6, f"  {key}:")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def add_image_centered(self, path, w=160):
        if os.path.exists(path):
            self.image(path, x=(self.w - w) / 2, w=w)
            self.ln(3)


def generate_report(den, clf, args, dataset_stats, start_time):
    pdf = HeartLensReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Cover page ──────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "HeartLens AI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Automated Training Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7,
             f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"TensorFlow {tf.__version__}  |  "
             f"Python {sys.version.split()[0]}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    # ── Table of Contents ───────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    toc_items = [
        "1. Configuration",
        "2. Dataset Summary",
        "3. Denoiser Results",
        "4. Classifier Results",
        "5. Quantized Model Validation",
        "6. Model Size & Latency",
        "7. Notes & Caveats",
    ]
    for item in toc_items:
        pdf.cell(0, 8, f"  {item}", new_x="LMARGIN", new_y="NEXT")

    # ── 1. Configuration ─────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("1. Configuration")
    pdf.key_value("Training epochs", args.epochs)
    pdf.key_value("Max segments/class", args.max_per_class)
    pdf.key_value("Data directory", args.data_dir)
    pdf.key_value("Noise levels (SNR dB)", ", ".join(map(str, NOISE_LEVELS)))
    pdf.key_value("Validation noise (dB)", VAL_NOISE_SNR)
    pdf.key_value("Sequence length", f"{SEQ_LEN} samples ({SEQ_LEN/360:.1f}s)")
    pdf.key_value("Output directory", str(OUT_DIR))

    # ── 2. Dataset Summary ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("2. Dataset Summary")
    pdf.body_text(
        f"Source: MIT-BIH Arrhythmia Database ({len(RECORDS_LIST)} recordings, "
        f"360 Hz, 2 leads, ~30 min each). Only MLII lead is used. "
        f"Segments are {WINDOW_SEC}-second windows centered on R-peaks. "
        f"AF rhythm data (afdb) is trained separately via afdb_loader.py."
    )
    pdf.body_text(f"Total segments loaded: {dataset_stats['total']}")
    for c in range(NUM_CLASSES):
        pdf.key_value(f"  Class {c} ({CLASS_NAMES[c]})",
                       dataset_stats['per_class'].get(c, 0))

    if dataset_stats.get('no_data_classes'):
        pdf.set_text_color(180, 60, 30)
        pdf.body_text(
            "WARNING: Classes "
            f"{', '.join(map(str, dataset_stats['no_data_classes']))} "
            "have zero training data after removing pathological proxy "
            "annotations. These classifier outputs are inactive."
        )
        pdf.set_text_color(30, 30, 30)

    fig_path = FIGS_DIR / "class_distribution.png"
    if os.path.exists(fig_path):
        pdf.add_image_centered(str(fig_path), w=150)

    # ── 3. Denoiser Results ─────────────────────────────────────
    if den is not None:
        pdf.add_page()
        pdf.section_title("3. Denoiser - Conv1D Autoencoder")
        pdf.body_text(
            f"Architecture: Input({SEQ_LEN},1) -> Conv1D(16,15) -> MaxPool1D(2) -> "
            "Conv1D(8,15) -> MaxPool1D(2) -> UpSampling1D(2) -> Conv1D(8,15) -> "
            "UpSampling1D(2) -> Conv1D(1,15). "
            "Why Conv1D over LSTM: 3-5x faster on ESP32 TFLite Micro, "
            "3x smaller (~50 KB vs ~148 KB), no unrolled recurrence."
        )
        pdf.key_value("Test MSE", f"{den['test_loss']:.6f}")
        pdf.key_value("Test MAE", f"{den['test_mae']:.6f}")
        val_loss = den['history'].history.get('val_loss', [0])
        train_loss = den['history'].history.get('loss', [0])
        pdf.key_value("Best epoch",
                       (np.argmin(val_loss) + 1) if len(val_loss) > 0 else 1)
        pdf.key_value("Total epochs trained", len(train_loss))

        fig_path = FIGS_DIR / "denoiser_history.png"
        if os.path.exists(fig_path):
            pdf.add_image_centered(str(fig_path), w=160)

        fig_path = FIGS_DIR / "denoiser_samples.png"
        if os.path.exists(fig_path):
            pdf.add_image_centered(str(fig_path), w=160)

    # ── 4. Classifier Results ───────────────────────────────────
    pdf.add_page()
    pdf.section_title("4. Classifier - 1D-CNN")
    pdf.body_text(
        f"Architecture: Input({SEQ_LEN},1) -> Conv1D(32,5)+BN+MaxPool -> "
        "Conv1D(64,5)+BN+MaxPool -> Conv1D(128,5)+BN+MaxPool -> "
        f"GAP -> Dense(64)+Dropout(0.5) -> Dense({NUM_CLASSES}, Softmax)."
    )

    # Float32 metrics table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Float32 Performance (per-class F1):",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for i, name in enumerate(CLASS_NAMES):
        if i < len(clf['f1_float']):
            pdf.cell(0, 6, f"  {name}: {clf['f1_float'][i]:.4f}",
                     new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    macro_f1 = np.mean(clf['f1_float'])
    pdf.key_value("Macro F1 (float32)", f"{macro_f1:.4f}")

    report = clf['report_dict']
    if 'weighted avg' in report:
        pdf.key_value("Weighted F1 (float32)",
                       f"{report['weighted avg']['f1-score']:.4f}")
    if 'accuracy' in report:
        pdf.key_value("Accuracy (float32)",
                       f"{report['accuracy']:.4f}")

    fig_path = FIGS_DIR / "classifier_history.png"
    if os.path.exists(fig_path):
        pdf.add_image_centered(str(fig_path), w=160)

    fig_path = FIGS_DIR / "confusion_matrix.png"
    if os.path.exists(fig_path):
        pdf.add_image_centered(str(fig_path), w=140)

    # ── 5. Quantized Model Validation ────────────────────────────
    pdf.add_page()
    pdf.section_title("5. Quantized Model Validation")

    # Denoiser quantized size
    if den is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Denoiser:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.key_value("  TFLite int8 size", f"{den['tflite_size_kb']:.1f} KB")
        pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Classifier:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.key_value("  TFLite int8 size", f"{clf['tflite_size_kb']:.1f} KB")
    pdf.key_value("  Input quantization",
                   f"scale={clf['in_quant'][0]:.6f}, "
                   f"zero={clf['in_quant'][1]}")
    pdf.key_value("  Output quantization",
                   f"scale={clf['out_quant'][0]:.6f}, "
                   f"zero={clf['out_quant'][1]}")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Per-class F1: Float32 vs Quantized Int8:",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    deltas = []
    for i, name in enumerate(CLASS_NAMES):
        if i < len(clf['f1_float']) and i < len(clf['f1_quant']):
            d = clf['f1_float'][i] - clf['f1_quant'][i]
            deltas.append(d)
            sign = "+" if d >= 0 else ""
            pdf.cell(0, 6,
                     f"  {name}: {clf['f1_float'][i]:.4f} -> "
                     f"{clf['f1_quant'][i]:.4f} ({sign}{d:.4f})",
                     new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.key_value("Macro F1 delta", f"{np.mean(deltas):.4f}" if deltas else "N/A")

    fig_path = FIGS_DIR / "confusion_matrix_quantized.png"
    if os.path.exists(fig_path):
        pdf.add_image_centered(str(fig_path), w=140)

    # ── 6. Model Size & Latency Estimate ────────────────────────
    pdf.add_page()
    pdf.section_title("6. Model Size & Latency Estimate")
    total_kb = (den['tflite_size_kb'] if den else 0) + clf['tflite_size_kb']
    if den:
        pdf.key_value("Denoiser (Conv1D int8)", f"{den['tflite_size_kb']:.1f} KB")
    pdf.key_value("Classifier (1D-CNN int8)", f"{clf['tflite_size_kb']:.1f} KB")
    pdf.key_value("Total models size", f"{total_kb:.1f} KB")
    pdf.key_value("TFLite arena (shared)",
                  "120 KB (60 KB per model)")
    pdf.key_value("ESP32 free heap (typical)",
                  "~200 KB after OS + BSS")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Estimated Inference Latency (ESP32 @ 240 MHz):",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "Per-window (360 samples): Denoiser ~2-3 ms, Classifier ~1-2 ms.  "
        "Total per sliding window: ~5 ms.  "
        f"19 windows @ 50% overlap: ~95 ms total inference.  "
        f"Target: < 100 ms.  "
        "ADC sampling: 10 s (background on Core 0).  "
        "End-to-end cycle: ~20 s (10 s sampling + 100 ms inference + "
        "10 s display)."
    )

    pdf.key_value("Per-window inference", "~5 ms")
    pdf.key_value("Total inference (19 windows)", "~95 ms")
    pdf.key_value("Battery life (500 mAh)", "~6.5 hours")

    # ── 7. Notes & Caveats ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("7. Notes & Caveats")
    notes = [
        "Regulatory: Research prototype. No FDA/CE clearance. Not "
        "intended to diagnose, treat, or manage any condition.",
        "Single-lead ECG: Limited diagnostic capability vs 12-lead. "
        "Cannot localise ST-elevation or determine axis.",
        "APB class: Class 1 is atrial premature beat (MIT-BIH symbol 'A'), "
        "honestly labeled. True AF rhythm detection requires afdb_loader.py "
        "training on the MIT-BIH Atrial Fibrillation Database.",
        "Patient-level CV recommended: Current split_dataset uses "
        "segment-level split which may leak data between train/test. "
        "Use split_by_patient() or group_kfold_eval.py for publication-"
        "grade results.",
        "ESP32 ADC: Internal ADC has ~6% non-linearity. For clinical-"
        "grade signal, use external ADC (ADS1292R).",
        "Per-buffer normalisation: Firmware applies [-1,1] scaling "
        "to the full 10 s buffer. Training normalises per 1 s window. "
        "Both use the same formula (center ÷ maxDev) but at different "
        "scopes - acceptable for stationary ECG.",
        "Latency and battery figures in this report are estimates; "
        "measure on hardware via BENCHMARK_MODE / REPLAY_MODE.",
        "This report was generated automatically by auto_train.py.",
    ]
    for note in notes:
        pdf.body_text(f"- {note}")

    # ── Save ─────────────────────────────────────────────────────
    report_path = OUT_DIR / "training_report.pdf"
    pdf.output(str(report_path))
    print(f"\n  PDF report: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════
#  GUI MODE
# ═══════════════════════════════════════════════════════════════════

class _TeeStream:
    """Writes to both a tkinter Text widget and the real stdout."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.real_stdout = sys.stdout

    def write(self, s):
        self.real_stdout.write(s)
        self.real_stdout.flush()
        if self.text_widget and s.strip():
            def append():
                self.text_widget.insert(tk.END, s)
                self.text_widget.see(tk.END)
            try:
                self.text_widget.after(0, append)
            except Exception:
                pass

    def flush(self):
        self.real_stdout.flush()


class TrainingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HeartLens AI - Training Pipeline")
        self.root.geometry("780x700")
        self.root.minsize(640, 500)
        self._build_ui()
        self._running = False
        self._thread = None

    def _build_ui(self):
        # ── Header ──
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill=tk.X)
        ttk.Label(header, text="HeartLens AI",
                  font=("Helvetica", 16, "bold")).pack(anchor=tk.W)
        ttk.Label(header, text="Automated Training & PDF Report Pipeline",
                  font=("Helvetica", 10)).pack(anchor=tk.W)

        # ── Parameters ──
        pf = ttk.LabelFrame(self.root, text="Parameters", padding=8)
        pf.pack(fill=tk.X, padx=10, pady=(0, 5))

        row = 0
        ttk.Label(pf, text="Data directory:").grid(row=row, column=0, sticky=tk.W, padx=(0, 8))
        self._data_dir = ttk.Entry(pf, width=40)
        self._data_dir.insert(0, "./mitdb")
        self._data_dir.grid(row=row, column=1, sticky=tk.EW)
        pf.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(pf, text="Epochs:").grid(row=row, column=0, sticky=tk.W, padx=(0, 8))
        self._epochs = ttk.Entry(pf, width=12)
        self._epochs.insert(0, "30")
        self._epochs.grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(pf, text="Max segments per class:").grid(row=row, column=0, sticky=tk.W, padx=(0, 8))
        self._max_per_class = ttk.Entry(pf, width=12)
        self._max_per_class.insert(0, "3000")
        self._max_per_class.grid(row=row, column=1, sticky=tk.W)
        row += 1

        self._skip_den = tk.BooleanVar(value=False)
        ttk.Checkbutton(pf, text="Skip denoiser", variable=self._skip_den).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        self._skip_clf = tk.BooleanVar(value=False)
        ttk.Checkbutton(pf, text="Skip classifier", variable=self._skip_clf).grid(row=row, column=0, columnspan=2, sticky=tk.W)

        # ── Progress ──
        self._progress = ttk.Progressbar(self.root, mode="indeterminate", length=760)
        self._progress.pack(fill=tk.X, padx=10, pady=(5, 0))

        # ── Log ──
        ttk.Label(self.root, text="Log output:", font=("Helvetica", 9)).pack(anchor=tk.W, padx=12, pady=(8, 0))
        self._log = scrolledtext.ScrolledText(self.root, height=20, font=("Consolas", 9),
                                               bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        self._log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 8))

        # ── Buttons ──
        bf = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bf.pack(fill=tk.X)
        self._run_btn = ttk.Button(bf, text="Run Training", command=self._run)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._cancel_btn = ttk.Button(bf, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT)

    def _log_msg(self, msg):
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)

    def _run(self):
        if self._running:
            return
        self._running = True
        self._run_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._log.delete(1.0, tk.END)
        self._progress.start(10)

        args = argparse.Namespace(
            data_dir=self._data_dir.get().strip() or "./mitdb",
            epochs=int(self._epochs.get().strip() or 30),
            max_per_class=int(self._max_per_class.get().strip() or 3000),
            skip_denoiser=self._skip_den.get(),
            skip_classifier=self._skip_clf.get(),
        )
        self._thread = threading.Thread(target=self._run_training, args=(args,), daemon=True)
        self._thread.start()
        self.root.after(500, self._poll_thread)

    def _run_training(self, args):
        old_stdout = sys.stdout
        try:
            sys.stdout = _TeeStream(self._log)
            main_inner(args)
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self._running = False

    def _poll_thread(self):
        if self._running:
            self.root.after(500, self._poll_thread)
        else:
            self._progress.stop()
            self._run_btn.config(state=tk.NORMAL)
            self._cancel_btn.config(state=tk.DISABLED)
            self._log_msg("\n[Done]")

    def _cancel(self):
        self._running = False
        self._progress.stop()
        self._run_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        self._log_msg("\n[Cancelled by user]")

    def run(self):
        self.root.mainloop()


def main_inner(args):
    """Core training logic (shared between CLI and GUI modes)."""
    os.makedirs(FIGS_DIR, exist_ok=True)
    start_time = datetime.datetime.now()

    print("=" * 60)
    print("  HeartLens AI \u2014 Automated Training Pipeline")
    print("=" * 60)
    print(f"  Data:       {args.data_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Max/class:  {args.max_per_class}")

    # ── Dataset summary ──────────────────────────────────────────
    print("\n  Loading dataset statistics...")
    by_class = load_all_segments(args.data_dir, max_per_class=args.max_per_class)
    dataset_stats = {"total": sum(len(v) for v in by_class.values()),
                     "per_class": {c: len(v) for c, v in by_class.items()}}
    no_data = [c for c, v in dataset_stats["per_class"].items() if v == 0]
    dataset_stats["no_data_classes"] = no_data
    print(f"  Total: {dataset_stats['total']} segments")
    for c in range(NUM_CLASSES):
        print(f"    Class {c} ({CLASS_NAMES[c]}): "
              f"{dataset_stats['per_class'].get(c, 0)}")

    # Class distribution figure (reconstruct splits for visualisation)
    (X_tr, y_tr), (X_vl, y_vl), (X_ts, y_ts) = split_dataset(by_class)
    plot_class_distribution(y_tr, y_vl, y_ts, CLASS_NAMES,
                             FIGS_DIR / "class_distribution.png")

    # ── Train denoiser ──────────────────────────────────────────
    if args.skip_denoiser:
        print("\n  Skipping denoiser (--skip-denoiser)")
        den = None
    else:
        den = train_denoiser(args.data_dir, args.max_per_class, args.epochs)
        plot_training_history(den['history'], "Denoiser",
                               FIGS_DIR / "denoiser_history.png")
        plot_denoiser_samples(den['X_test'], den['X_test_noisy'],
                               den['model'],
                               FIGS_DIR / "denoiser_samples.png")

    # ── Train classifier ─────────────────────────────────────────
    if args.skip_classifier:
        print("\n  Skipping classifier (--skip-classifier)")
        clf = None
    else:
        clf = train_classifier(args.data_dir, args.max_per_class, args.epochs)
        plot_training_history(clf['history'], "Classifier",
                               FIGS_DIR / "classifier_history.png")
        plot_confusion_matrix(clf['cm'], CLASS_NAMES,
                               "Confusion Matrix (float32)",
                               FIGS_DIR / "confusion_matrix.png")
        plot_confusion_matrix(clf['cm_q'], CLASS_NAMES,
                               "Confusion Matrix (quantized int8)",
                               FIGS_DIR / "confusion_matrix_quantized.png")

    # ── Generate PDF report ──────────────────────────────────────
    if den is None and clf is None:
        print("\n  Nothing to report (both models skipped). Exiting.")
        return

    print("\n" + "=" * 60)
    print("  Generating PDF report...")
    report_path = generate_report(den, clf, args, dataset_stats, start_time)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  Report: {report_path}")
    if den:
        print(f"  Denoiser:   {den['tflite_size_kb']:.1f} KB TFLite, "
              f"MAE={den['test_mae']:.6f}")
    if clf:
        print(f"  Classifier: {clf['tflite_size_kb']:.1f} KB TFLite, "
              f"Macro F1={np.mean(clf['f1_float']):.4f}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HeartLens AI - Automated Training and PDF Report"
    )
    parser.add_argument("--data-dir", default="./mitdb",
                        help="MIT-BIH database directory")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs")
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="Max segments per class")
    parser.add_argument("--skip-denoiser", action="store_true",
                        help="Skip denoiser training (use existing)")
    parser.add_argument("--skip-classifier", action="store_true",
                        help="Skip classifier training (use existing)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch graphical interface")
    args = parser.parse_args()

    if args.gui:
        if not _HAVE_TK:
            print("Tkinter is not available on this system. "
                  "Install python3-tk (Ubuntu/Debian) or use CLI mode.")
            sys.exit(1)
        app = TrainingGUI()
        app.run()
        return

    main_inner(args)


if __name__ == "__main__":
    main()
