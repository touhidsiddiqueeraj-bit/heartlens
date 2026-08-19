#!/usr/bin/env python3
"""Experiment 2 — Noise robustness: Raw+CNN vs Filter+CNN vs AE+CNN.

At each SNR in {0,5,10,15,20,30,40} dB, the test set is corrupted and
classified through three front-ends. Demonstrates whether the learned
autoencoder denoiser actually improves classification (review #7).

Usage:
    python3 evaluate_noise_robustness.py --epochs 40
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import signal as sg
from sklearn.metrics import f1_score

from data_loader import WINDOW_SAMPLES, NUM_CLASSES
from noise_pipeline import add_all_noise
from models import build_classifier

OUT_DIR = Path(__file__).parent / "results"
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_LEVELS = (0, 5, 10, 15, 20, 30, 40)
CLASS_NAMES = ["Normal", "APB", "PVC"]


def build_denoiser():
    # Decoder upsamples with Conv1DTranspose (=> TRANSPOSE_CONV in TFLite,
    # supported by TFLM) instead of UpSampling1D, which Keras 3 quantizes to
    # a TILE op that TFLM builds for ESP32 don't include.
    inputs = tf.keras.layers.Input(shape=(WINDOW_SAMPLES, 1), name="denoiser_input")
    x = tf.keras.layers.Conv1D(16, 15, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(8, 15, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1DTranspose(8, 15, strides=2, padding="same",
                                        activation="relu")(x)
    x = tf.keras.layers.Conv1D(8, 15, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1DTranspose(8, 15, strides=2, padding="same",
                                        activation="relu")(x)
    x = tf.keras.layers.Conv1D(1, 15, padding="same")(x)
    model = tf.keras.Model(inputs, x, name="denoiser")
    model.compile(optimizer="adam", loss="mse")
    return model


def bandpass_filter(X):
    sos = sg.butter(4, [0.5, 45], btype="band", fs=360, output="sos")
    out = np.stack([sg.sosfilt(sos, x[:, 0]) for x in X])
    return out.reshape(-1, WINDOW_SAMPLES, 1)


def macro_f1(y_true, y_pred):
    return float(np.mean(f1_score(y_true, y_pred, average=None, zero_division=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./mitdb")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models-dir", default="./models",
                    help="dir to cache trained classifier/denoiser")
    args = ap.parse_args()

    from train_classifier import load_data_with_record_tracking, record_level_split
    by_class = load_data_with_record_tracking(args.data_dir, args.max_per_class)
    (X_tr, y_tr), _, (X_te, y_te) = record_level_split(by_class)
    X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_te = X_te.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    print(f"Train {X_tr.shape}, Test {X_te.shape}")
    print(f"y bincounts  train={np.bincount(y_tr)} test={np.bincount(y_te)}")

    os.makedirs(args.models_dir, exist_ok=True)
    clf_path = os.path.join(args.models_dir, "robust_classifier.keras")
    den_path = os.path.join(args.models_dir, "robust_denoiser.keras")

    # ── Train / load classifier ───────────────────────────────────
    if os.path.exists(clf_path):
        clf = tf.keras.models.load_model(clf_path)
    else:
        clf = build_classifier()
        clf.fit(X_tr, y_tr, epochs=args.epochs, batch_size=64,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,
                                                            restore_best_weights=True)],
                verbose=0)
        clf.save(clf_path)

    # ── Train / load denoiser (clean reconstruction) ──────────────
    if os.path.exists(den_path):
        den = tf.keras.models.load_model(den_path)
    else:
        den = build_denoiser()
        noisy = np.stack([add_all_noise(x.flatten(), snr_db=15) for x in X_tr])
        den.fit(noisy.reshape(-1, WINDOW_SAMPLES, 1), X_tr,
                epochs=args.epochs, batch_size=64, verbose=0)
        den.save(den_path)

    # ── Evaluate ──────────────────────────────────────────────────
    results = {pipe: {} for pipe in ("raw", "filter", "autoencoder")}
    for snr in NOISE_LEVELS:
        X_noisy = np.stack([add_all_noise(x.flatten(), snr_db=snr) for x in X_te])
        X_noisy = X_noisy.reshape(-1, WINDOW_SAMPLES, 1)

        pred_raw = np.argmax(clf.predict(X_noisy, verbose=0), axis=1)
        results["raw"][snr] = macro_f1(y_te, pred_raw)

        pred_filt = np.argmax(clf.predict(bandpass_filter(X_noisy), verbose=0), axis=1)
        results["filter"][snr] = macro_f1(y_te, pred_filt)

        pred_ae = np.argmax(clf.predict(den.predict(X_noisy, verbose=0), verbose=0), axis=1)
        results["autoencoder"][snr] = macro_f1(y_te, pred_ae)

        print(f"SNR={snr:2d} dB  raw={results['raw'][snr]:.4f}  "
              f"filter={results['filter'][snr]:.4f}  "
              f"ae={results['autoencoder'][snr]:.4f}")

    # ── Plot ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for pipe in results:
        ax.plot(NOISE_LEVELS, [results[pipe][s] for s in NOISE_LEVELS],
                marker="o", label={"raw": "Raw + CNN",
                                   "filter": "Bandpass + CNN",
                                   "autoencoder": "Autoencoder + CNN"}[pipe])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Noise Robustness: Raw vs Filter vs Autoencoder front-end")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_path = OUT_DIR / "noise_robustness.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    with open(OUT_DIR / "noise_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {fig_path}, {OUT_DIR / 'noise_robustness.json'}")


if __name__ == "__main__":
    main()
