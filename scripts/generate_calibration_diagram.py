#!/usr/bin/env python3
"""Calibration reliability diagram — canonical version.

Protocol (matches calibrate.py / paper claim):
- Model: saved robust_classifier.keras (NO retraining)
- Split: frozen record-level split (seed=42); T fit on VAL by calibrate.py
- Metrics: ECE/NLL/Brier before vs after temperature scaling on held-out TEST
Outputs: results/calibration_extended.json (updated) + results/calibration_fig.png
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / "heart-lens-training"))
from train_classifier import load_data_with_record_tracking, record_level_split  # noqa: E402

ROOT = Path(__file__).parent.parent
MODELS = ROOT / "heart-lens-training" / "models"
OUT = ROOT / "heart-lens-training" / "results"
N_BINS = 10

model = tf.keras.models.load_model(MODELS / "robust_classifier.keras")
CAL = json.loads((OUT / "calibration.json").read_text())
T = float(CAL["temperature"])
print(f"robust_classifier.keras loaded; T={T:.4f} (fit on val by calibrate.py)")

by_class = load_data_with_record_tracking(str(ROOT / "heart-lens-training" / "mitdb"), 3000)
(_, _), (_, _), (X_te, y_te) = record_level_split(by_class)
X_te = X_te.reshape(-1, 360, 1).astype(np.float32)
n = len(y_te)
print(f"held-out test: n={n}, bincount={np.bincount(y_te)}")

logits = model.predict(X_te, verbose=0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, labels):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    total = len(labels)
    s = 0.0
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        s += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(s)


def nll(probs, labels):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1))))


def brier(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


probs_raw = softmax(logits)
probs_cal = softmax(logits / T)

metrics = {
    "temperature": T,
    "split": "test (frozen record-level 493/24/401); T fit on val by calibrate.py",
    "n": int(n),
    "ece_raw": ece(probs_raw, y_te), "ece_cal": ece(probs_cal, y_te),
    "nll_raw": nll(probs_raw, y_te), "nll_cal": nll(probs_cal, y_te),
    "brier_raw": brier(probs_raw, y_te), "brier_cal": brier(probs_cal, y_te),
    "bins": N_BINS,
}
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2))

(OUT / "calibration_extended.json").write_text(json.dumps(metrics, indent=2))

# ---- reliability diagram ----
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, probs, ttl in [
    (axes[0], probs_raw, f"Before\nECE={metrics['ece_raw']:.3f}"),
    (axes[1], probs_cal, f"After T={T:.2f}\nECE={metrics['ece_cal']:.3f}"),
]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_te).astype(float)
    xs, accs, fracs = [], [], []
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean()); accs.append(correct[m].mean()); fracs.append(m.mean())
    ax.bar(xs, accs, width=0.09, alpha=0.35, color="#1f77b4", edgecolor="none")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="Perfect")
    ax.plot(xs, accs, "o-", color="#1f77b4", ms=3.5, lw=1.2, label="Reliability")
    ax.set_title(ttl, fontsize=8.5)
    ax.set_xlabel("Confidence", fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Accuracy", fontsize=7.5)
axes[0].legend(fontsize=6.5, loc="upper left")
fig.suptitle(f"Reliability Diagram  (test n={n}, T={T:.2f})", fontsize=9, weight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out_png = OUT / "calibration_fig.png"
fig.savefig(out_png, dpi=300)
print(f"saved {out_png}")
