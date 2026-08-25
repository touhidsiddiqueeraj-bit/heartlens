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

# ---- reliability diagrams: two single-column panels (a) Before and (b) After, stacked vertically in paper ----
# Single-column width = 3.4in, each panel ~2.2in tall, readable at 100% zoom
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
for suffix, probs, ttl, col in [
    ("a_before", probs_raw, f"(a) Before — ECE {metrics['ece_raw']:.3f}", "#c0392b"),
    ("b_after", probs_cal, f"(b) After T={T:.2f} — ECE {metrics['ece_cal']:.3f}", "#2471a3"),
]:
    fig, ax = plt.subplots(figsize=(3.40, 2.35))
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
    alphas = [0.28 + 0.62 * (f / max(fracs)) for f in fracs]
    for x, a, al in zip(xs, accs, alphas):
        ax.bar(x, a, width=0.085, alpha=al, color=col, edgecolor="white", linewidth=0.6, zorder=2)
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.2, dashes=(4, 3), label="Perfect")
    ax.plot(xs, accs, "o-", color=col, ms=4.5, lw=1.6, markeredgecolor="white", markeredgewidth=0.8, label="Reliability", zorder=3)
    ax.set_title(ttl, fontsize=8.0, pad=7, weight="bold", color="#1a1a1a")
    ax.set_xlabel("Confidence", fontsize=7.5)
    ax.set_ylabel("Accuracy", fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=7, width=0.6, length=3, direction="out", pad=2.2)
    ax.grid(alpha=0.15, linewidth=0.5, linestyle="-")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6); spine.set_color("#444")
    ax.legend(fontsize=6.2, loc="upper left", frameon=True, facecolor="white", edgecolor="#bbb", framealpha=0.95, handlelength=1.2, borderpad=0.28, labelspacing=0.22)
    fig.tight_layout()
    fig.subplots_adjust(top=0.87, bottom=0.15, left=0.13, right=0.97)
    out_png = OUT / f"calibration_{suffix}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"saved {out_png}  {out_png.stat().st_size/1024:.0f}KB")
# keep combined for backward compat (not used in paper)
out_png = OUT / "calibration_fig.png"
print(f"combined legacy {out_png} kept for compat")
