#!/usr/bin/env python3
"""Experiment 3 — External generalization.

Train the 3-class beat classifier on MIT-BIH (patient-level split),
evaluate WITHOUT retraining on:
  - MIT-BIH SVDB (supraventricular arrhythmia DB, 128 Hz): beat-level N/A/V
  - MIT-BIH afdb: rhythm windows; report the class distribution the
    beat-level model assigns to true AF rhythm (interpretation only —
    AF windows are NOT ground-truth beats for this model).

Usage:
    python3 external_validation.py --epochs 40
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import signal as sg
from sklearn.metrics import classification_report, f1_score

from data_loader import WINDOW_SAMPLES, NUM_CLASSES
from models import build_classifier

OUT_DIR = Path(__file__).parent / "results"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "APB", "PVC"]

SVDB_RECORDS = [800, 801, 802, 803, 804, 805, 806, 807, 808, 809,
                810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820]
SVDB_RATE = 128
SVDB_SYM_TO_CLASS = {'N': 0, 'A': 1, 'V': 2}


def _normalize(seg):
    center = np.mean(seg)
    dev = np.max(np.abs(seg - center))
    return (seg - center) / dev if dev > 1e-12 else seg - center


def load_svdb(data_dir="./svdb"):
    import wfdb
    if not os.path.exists(data_dir) or not any(
            f.endswith('.dat') for f in os.listdir(data_dir)):
        if os.path.exists(data_dir):
            import shutil
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        wfdb.dl_database('svdb', data_dir,
                         records=[str(r) for r in SVDB_RECORDS])

    X, y = [], []
    for rec in os.listdir(data_dir):
        if not rec.endswith('.dat'):
            continue
        name = rec[:-4]
        try:
            record = wfdb.rdrecord(os.path.join(data_dir, name))
            ann = wfdb.rdann(os.path.join(data_dir, name), 'atr')
        except Exception as e:
            print(f"  {name}: SKIP ({e})")
            continue
        sig = record.p_signal[:, 0]
        sig = sg.resample_poly(sig, 360, SVDB_RATE)  # 128 -> 360 Hz
        n = 0
        for peak, sym in zip(ann.sample, ann.symbol):
            cls = SVDB_SYM_TO_CLASS.get(sym, None)
            if cls is None:
                continue
            peak = int(peak * 360 / SVDB_RATE)
            start = peak - WINDOW_SAMPLES // 2
            end = start + WINDOW_SAMPLES
            if start < 0 or end >= len(sig):
                continue
            X.append(_normalize(sig[start:end]))
            y.append(cls)
            n += 1
        print(f"  {name}: {n} segments")
    return np.array(X).reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32), np.array(y)


def load_afdb_windows(data_dir="./afdb", max_windows=1000):
    """AF rhythm windows for distribution check (not label evaluation)."""
    from afdb_loader import load_afdb_rhythm
    X, y = load_afdb_rhythm(data_dir, max_per_class=max_windows)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitdb-dir", default="./mitdb")
    ap.add_argument("--svdb-dir", default="./svdb")
    ap.add_argument("--afdb-dir", default="./afdb")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    # ── Train on MIT-BIH with patient-level split ──────────────────
    from train_classifier import load_data_with_record_tracking, record_level_split
    by_class = load_data_with_record_tracking(args.mitdb_dir, args.max_per_class)
    (X_tr, y_tr), _, _ = record_level_split(by_class, train_ratio=0.85, val_ratio=0.15)
    X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    print(f"Train: {X_tr.shape}, {np.bincount(y_tr)}")

    model = build_classifier()
    model.fit(X_tr, y_tr, epochs=args.epochs, batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,
                                                          restore_best_weights=True)],
              verbose=0)
    model.save(OUT_DIR / "external_model.keras")

    # ── Evaluate on SVDB ───────────────────────────────────────────
    print("\n=== SVDB (external beat-level database) ===")
    X_svdb, y_svdb = load_svdb(args.svdb_dir)
    pred = np.argmax(model.predict(X_svdb, verbose=0), axis=1)
    print(classification_report(y_svdb, pred, target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    f1_svdb = f1_score(y_svdb, pred, average=None, zero_division=0)
    print(f"Per-class F1 on SVDB: {np.round(f1_svdb, 4)}")

    # ── AF rhythm distribution check (interpretation only) ─────────
    print("\n=== afdb rhythm windows: predicted class distribution ===")
    X_af, y_af = load_afdb_windows(args.afdb_dir)
    pred_af = np.argmax(model.predict(X_af, verbose=0), axis=1)
    for cls, name in enumerate(CLASS_NAMES):
        frac = np.mean(pred_af[y_af == 1] == cls)
        print(f"  AF windows predicted as {name:6s}: {frac:.1%}")
    for cls, name in enumerate(CLASS_NAMES):
        frac = np.mean(pred_af[y_af == 0] == cls)
        print(f"  Normal windows predicted as {name:6s}: {frac:.1%}")

    out = {"svdb_f1_per_class": f1_svdb.tolist(),
           "svdb_macro_f1": float(np.mean(f1_svdb)),
           "afdb_af_distribution": {
               name: float(np.mean(pred_af[y_af == 1] == cls))
               for cls, name in enumerate(CLASS_NAMES)},
           "afdb_normal_distribution": {
               name: float(np.mean(pred_af[y_af == 0] == cls))
               for cls, name in enumerate(CLASS_NAMES)}}
    path = OUT_DIR / "external_validation.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
