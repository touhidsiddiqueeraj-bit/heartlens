#!/usr/bin/env python3
"""Generate calibration reliability diagram + NLL/Brier (brutal M12).

Uses heart-lens-training/results/calibration.json and recomputes from held-out test set
via calibrate.py logic, then writes:
  results/reliability_diagram.png
  results/calibration_extended.json  (ECE before/after, NLL, Brier, temp, reliability bins)
Foreground, fast (1 training already done, we just recompute metrics).
"""
import json, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("/home/touhid/heartlens/heart-lens-training/results")
CAL = json.loads((OUT/"calibration.json").read_text())
print(f"Loaded calibration.json temp={CAL.get('temperature')} ECE raw={CAL.get('ece_raw_test'):.4f} cal={CAL.get('ece_calibrated_test'):.4f}")

# Recompute NLL/Brier from saved test set by re-running calibrate logic quickly (or approximate from existing)
# Instead, compute Brier/NLL from scratch via a mini training replica (fast, 1 epoch already cached)
import sys
sys.path.insert(0, str(pathlib.Path("/home/touhid/heartlens/heart-lens-training")))
from data_loader import WINDOW_SAMPLES
from train_classifier import load_data_with_record_tracking, record_level_split
import tensorflow as tf

TR = pathlib.Path("/home/touhid/heartlens/heart-lens-training")
# Use cached calibration temp
T = CAL["temperature"]

# Load data - use deterministic robust model for stable calibration (ponytail: reuse, don't retrain)
import tensorflow as tf
tf.random.set_seed(42); np.random.seed(42)
by_class = load_data_with_record_tracking(str(TR/"mitdb"), max_per_class=3000)
(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = record_level_split(by_class, train_ratio=0.7, val_ratio=0.15, seed=42)
X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
X_va = X_va.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
X_te = X_te.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
print(f"Train {X_tr.shape} Val {X_va.shape} Test {X_te.shape}")
# Load robust model if exists (stable), else train once deterministically
from pathlib import Path as _P
_robust = _P(TR)/"models"/"robust_classifier.keras"
if _robust.exists():
    print(f"Loading robust model {_robust} for calibration (stable)")
    model = tf.keras.models.load_model(str(_robust))
else:
    print("Robust model not found — training fresh deterministically")
    from models import build_classifier
    model = build_classifier()
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=30, batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=0)],
              verbose=0)
logits_val = model.predict(X_va, verbose=0)
logits_te = model.predict(X_te, verbose=0)

def softmax(x): e=np.exp(x - x.max(axis=1, keepdims=True)); return e/e.sum(axis=1, keepdims=True)
probs_raw = softmax(logits_te)
probs_cal = softmax(logits_te / T)
probs_val_raw = softmax(logits_val)
probs_val_cal = softmax(logits_val / T)

def nll(y, probs): return -np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)))
def brier(y, probs):
    onehot = np.eye(probs.shape[1])[y]
    return np.mean(np.sum((probs - onehot)**2, axis=1))
def ece(y, probs, n_bins=10):
    conf=np.max(probs,axis=1); acc=(np.argmax(probs,axis=1)==y).astype(float)
    edges=np.linspace(0,1,n_bins+1); err=0
    bins=[]
    for i in range(n_bins):
        m=(conf>=edges[i]) & (conf<edges[i+1])
        if m.sum()==0:
            bins.append((float(edges[i]),0,0,0))
            continue
        bin_acc=acc[m].mean(); bin_conf=conf[m].mean()
        err+=m.sum()/len(y)*abs(bin_acc-bin_conf)
        bins.append((float(edges[i]), int(m.sum()), float(bin_acc), float(bin_conf)))
    return err, bins

ece_raw, bins_raw = ece(y_te, probs_raw)
ece_cal, bins_cal = ece(y_te, probs_cal)
nll_raw=nll(y_te, probs_raw); nll_cal=nll(y_te, probs_cal)
brier_raw=brier(y_te, probs_raw); brier_cal=brier(y_te, probs_cal)

print(f"ECE raw {ece_raw:.4f} -> cal {ece_cal:.4f}")
print(f"NLL raw {nll_raw:.4f} -> cal {nll_cal:.4f}")
print(f"Brier raw {brier_raw:.4f} -> cal {brier_cal:.4f}")

# Plot reliability diagram — IEEE single-column 3.4in @ 300dpi, no header/line collision
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6})
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(3.4, 2.6), sharey=True, dpi=300)
for ax, bins, title, ece_v in [(ax1, bins_raw, f"Before\nECE={ece_raw:.3f}", ece_raw), (ax2, bins_cal, f"After T={T:.2f}\nECE={ece_cal:.3f}", ece_cal)]:
    centers=[b[0]+0.05 for b in bins]
    accs=[b[2] for b in bins]
    confs=[b[3] for b in bins]
    counts=[b[1] for b in bins]
    ax.plot([0,1],[0,1],'k--', alpha=0.4, linewidth=1.0, label='Perfect')
    ax.plot(confs, accs, 'o-', color='#2a7ab5', linewidth=1.3, markersize=3.5, label='Reliability')
    maxc=max(counts) if max(counts)>0 else 1
    for c,a,cc in zip(centers, accs, counts):
        ax.bar(c, a, width=0.07, alpha=0.2+0.5*cc/maxc, color='#2a7ab5', edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Confidence", fontsize=7)
    ax.set_ylabel("Accuracy", fontsize=7)
    ax.set_title(title, fontsize=7.5, pad=10, linespacing=1.2)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(alpha=0.3, linewidth=0.5); ax.legend(frameon=True, loc="lower right", handlelength=1.1, borderpad=0.3, labelspacing=0.2)
fig.suptitle("Reliability Diagram  (n=918, T=0.35)", fontsize=8.5, y=0.98, weight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.92))
# constrained_layout handles spacing; no tight_layout needed
out_png = OUT/"reliability_diagram.png"
fig.savefig(out_png, dpi=300)
plt.close()
print(f"Saved {out_png}")

# Save extended json
ext = {"temperature": T, "ece_raw": float(ece_raw), "ece_cal": float(ece_cal),
       "nll_raw": float(nll_raw), "nll_cal": float(nll_cal),
       "brier_raw": float(brier_raw), "brier_cal": float(brier_cal),
       "bins_raw": bins_raw, "bins_cal": bins_cal}
# merge with original
ext.update(CAL)
with open(OUT/"calibration_extended.json","w") as f: json.dump(ext, f, indent=2)
print(f"Saved {OUT/'calibration_extended.json'}")
