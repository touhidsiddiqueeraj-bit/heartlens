#!/usr/bin/env python3
"""Experiment 1 — Patient-independent classification with grouped CV.

GroupKFold at the RECORD (patient) level, repeated over multiple seeds.
Reports mean +/- std per-class and macro F1 (review #8).

Usage:
    python3 group_kfold_eval.py --folds 5 --seeds 0,1,2 --epochs 40
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from data_loader import WINDOW_SAMPLES, NUM_CLASSES
from models import build_classifier

OUT_DIR = Path(__file__).parent / "results"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "APB", "PVC"]


def load_arrays(data_dir, max_per_class):
    """Return X (n, WINDOW_SAMPLES, 1), y, groups from record-aware loading."""
    from train_classifier import load_data_with_record_tracking
    by_class = load_data_with_record_tracking(data_dir, max_per_class)
    X_all, y_all, g_all = [], [], []
    for cls, items in by_class.items():
        for seg, rec in items:
            X_all.append(seg)
            y_all.append(cls)
            g_all.append(rec)
    X = np.array(X_all).reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    return X, np.array(y_all), np.array(g_all)


def run_fold(X, y, groups, train_idx, test_idx, epochs):
    model = build_classifier()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4),
    ]
    model.fit(X[train_idx], y[train_idx],
              validation_data=(X[test_idx], y[test_idx]),
              epochs=epochs, batch_size=64,
              callbacks=callbacks, verbose=0)
    pred = np.argmax(model.predict(X[test_idx], verbose=0), axis=1)
    return f1_score(y[test_idx], pred, average=None, zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./mitdb")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    X, y, groups = load_arrays(args.data_dir, args.max_per_class)
    print(f"Loaded: X={X.shape}, classes={np.bincount(y)}, records={len(set(groups))}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_f1 = []  # (seed, fold, per-class F1 array)

    for seed in seeds:
        gkf = GroupKFold(n_splits=args.folds)
        # GroupKFold has no RNG; shuffle record groups deterministically per seed
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        perm = {r: i for i, r in enumerate(rng.permutation(uniq))}
        g_perm = np.array([perm[g] for g in groups])
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=g_perm)):
            f1 = run_fold(X, y, groups, tr, te, args.epochs)
            all_f1.append((seed, fold, f1))
            print(f"seed={seed} fold={fold}  "
                  f"per-class F1={np.round(f1, 4)}  macro={np.mean(f1):.4f}")

    all_f1 = np.array([f for _, _, f in all_f1])
    mean = all_f1.mean(axis=0)
    std = all_f1.std(axis=0)
    print("\n=== Grouped CV results ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:8s} F1 = {mean[i]:.4f} +/- {std[i]:.4f}")
    print(f"  Macro    F1 = {mean.mean():.4f} +/- {std.mean():.4f}")

    out = {"class_names": CLASS_NAMES,
           "mean": mean.tolist(), "std": std.tolist(),
           "macro_mean": float(mean.mean()), "macro_std": float(std.mean()),
           "folds": args.folds, "seeds": seeds}
    path = OUT_DIR / "group_kfold.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
