#!/usr/bin/env python3
"""Confidence calibration — temperature scaling (review #12).

Fits a single temperature T on the validation set to minimize negative
log-likelihood, reports ECE (expected calibration error) before/after,
and writes CALIB_TEMPERATURE into the firmware Config.h.

Usage:
    python3 calibrate.py --epochs 40
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar

from data_loader import WINDOW_SAMPLES
from models import build_classifier

OUT_DIR = Path(__file__).parent / "results"
os.makedirs(OUT_DIR, exist_ok=True)
CONFIG_H = Path(__file__).parent.parent / "HeartLens_Firmware" / "src" / "Config.h"


def ece(y_true, probs, n_bins=10):
    """Expected calibration error over confidence bins."""
    conf = np.max(probs, axis=1)
    acc = (np.argmax(probs, axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    err, n = 0.0, 0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        err += m.sum() / len(y_true) * abs(acc[m].mean() - conf[m].mean())
        n += 1
    return err


def nll_with_temperature(t, logits, y_true):
    logits_t = logits / t
    logsm = tf.nn.log_softmax(logits_t, axis=1).numpy()
    return -np.mean(logsm[np.arange(len(y_true)), y_true])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./mitdb")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--write-config", action="store_true",
                    help="write CALIB_TEMPERATURE into firmware Config.h")
    args = ap.parse_args()

    from train_classifier import load_data_with_record_tracking, record_level_split
    by_class = load_data_with_record_tracking(args.data_dir, args.max_per_class)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = record_level_split(by_class)
    X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_va = X_va.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_te = X_te.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    print(f"Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")

    model = build_classifier()
    model.fit(X_tr, y_tr, epochs=args.epochs, batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,
                                                          restore_best_weights=True)],
              verbose=0)

    logits_val = model.predict(X_va, verbose=0)
    probs_val = tf.nn.softmax(logits_val, axis=1).numpy()
    y_pred = np.argmax(probs_val, axis=1)

    ece_before = ece(y_va, probs_val)
    acc = np.mean(y_pred == y_va)
    print(f"\nECE before calibration: {ece_before:.4f}  (accuracy={acc:.4f})")

    res = minimize_scalar(
        lambda t: nll_with_temperature(t, logits_val, y_va),
        bounds=(0.1, 10.0), method="bounded")
    T = float(res.x)
    print(f"Optimal temperature T = {T:.4f}")

    # Apply to test set and re-measure ECE
    logits_te = model.predict(X_te, verbose=0)
    probs_te = tf.nn.softmax(logits_te / T, axis=1).numpy()
    ece_after = ece(y_te, probs_te)
    ece_raw = ece(y_te, tf.nn.softmax(logits_te, axis=1).numpy())
    print(f"Test ECE raw={ece_raw:.4f}  calibrated={ece_after:.4f}")

    # Reliability table
    print("\nReliability table (calibrated, test set):")
    conf = np.max(probs_te, axis=1)
    acc_bin = (np.argmax(probs_te, axis=1) == y_te).astype(float)
    for lo in np.linspace(0, 0.9, 10):
        m = (conf >= lo) & (conf < lo + 0.1)
        if m.sum():
            print(f"  [{lo:.1f},{lo+0.1:.1f}): n={m.sum():5d}  "
                  f"mean_conf={conf[m].mean():.3f}  acc={acc_bin[m].mean():.3f}")

    out = {"temperature": T, "ece_raw_test": float(ece_raw),
           "ece_calibrated_test": float(ece_after)}
    with open(OUT_DIR / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT_DIR / 'calibration.json'}")

    if args.write_config and CONFIG_H.exists():
        text = CONFIG_H.read_text()
        text = re.sub(r"#define CALIB_TEMPERATURE\s+[\d.]+f",
                      f"#define CALIB_TEMPERATURE   {T:.4f}f", text)
        CONFIG_H.write_text(text)
        print(f"Wrote CALIB_TEMPERATURE = {T:.4f} to {CONFIG_H}")


if __name__ == "__main__":
    main()
