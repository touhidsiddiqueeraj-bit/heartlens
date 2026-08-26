#!/usr/bin/env python3
"""Round-3 P3 — Calibration evaluated on the deployed INT8 model (PC-side).

Loads the saved robust_classifier.keras (NO retraining), quantizes it with the
same PTQ recipe as Table V (200 train-split samples, full-int8), then reports
ECE/NLL/Brier for: FP32 raw, FP32 + T_fp32 (existing val-fit temperature),
INT8 raw, INT8 + T_int8 (fresh temperature fit on INT8 validation outputs).

Resumable: writes results/calibration_int8.json atomically; skips if present.
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import tensorflow as tf

TR = Path(__file__).resolve().parents[1] / "heart-lens-training"
sys.path.insert(0, str(TR))
OUT = TR / "results"
N_BINS = 10
DEST = OUT / "calibration_int8.json"

if DEST.exists():
    try:
        if json.loads(DEST.read_text()).get("complete"):
            print(f"[int8-calib] skip (exists): {DEST}")
            sys.exit(0)
    except Exception:
        pass

from train_classifier import load_data_with_record_tracking, record_level_split  # noqa: E402

model = tf.keras.models.load_model(TR / "models" / "robust_classifier.keras")
by_class = load_data_with_record_tracking(str(TR / "mitdb"), 3000)
(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = record_level_split(by_class)
X_tr = X_tr.reshape(-1, 360, 1).astype(np.float32)
X_va = X_va.reshape(-1, 360, 1).astype(np.float32)
X_te = X_te.reshape(-1, 360, 1).astype(np.float32)
print(f"train={len(y_tr)} val={len(y_va)} test={len(y_te)}", flush=True)

T_FP32 = float(json.loads((OUT / "calibration.json").read_text())["temperature"])


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, labels):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    s = 0.0
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.sum():
            s += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(s)


def nll(probs, labels):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1))))


def brier(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_T(probs_val, y_val):
    """Fit temperature on val by NLL minimization (logits = log p)."""
    logits = np.log(np.clip(probs_val, 1e-12, 1))
    def obj(t):
        p = softmax(logits / t)
        return nll(p, y_val)
    res = minimize_scalar(obj, bounds=(0.05, 5.0), method="bounded")
    return float(res.x)


# ---- FP32 side ----
logits_te = model.predict(X_te, verbose=0)
logits_va = model.predict(X_va, verbose=0)
p32_te_raw = softmax(logits_te); p32_va_raw = softmax(logits_va)
p32_te_T = softmax(logits_te / T_FP32)

# ---- INT8 side (same PTQ recipe as Table V) ----
idx = np.random.RandomState(0).choice(len(X_tr), min(200, len(X_tr)), replace=False)
def rep():
    for i in idx:
        yield [X_tr[i:i + 1]]
c = tf.lite.TFLiteConverter.from_keras_model(model)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.int8
c.inference_output_type = tf.int8
tflite = c.convert()
print(f"int8 tflite size {len(tflite)/1024:.1f} KB", flush=True)

interp = tf.lite.Interpreter(model_content=tflite)
interp.allocate_tensors()
ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
s_in, z_in = ind["quantization"]; s_out, z_out = outd["quantization"]

def int8_run(X):
    out = np.zeros((len(X), 3), dtype=np.float32)
    for i in range(len(X)):
        q = np.round(X[i] / s_in + z_in).clip(-128, 127).astype(np.int8)
        interp.set_tensor(ind["index"], q.reshape(1, 360, 1))
        interp.invoke()
        raw = interp.get_tensor(outd["index"])[0].astype(np.float32)
        out[i] = (raw - z_out) * s_out
    return out

p8_va_raw = int8_run(X_va)
p8_te_raw = int8_run(X_te)
T_INT8 = fit_T(p8_va_raw, y_va)
# temperature on prob outputs: logits = log p
p8_te_T = softmax(np.log(np.clip(p8_te_raw, 1e-12, 1)) / T_INT8)

res = {
    "complete": True,
    "n_val": int(len(y_va)), "n_test": int(len(y_te)),
    "T_fp32": T_FP32, "T_int8": round(T_INT8, 4),
    "fp32_raw": {"ece": ece(p32_te_raw, y_te), "nll": nll(p32_te_raw, y_te), "brier": brier(p32_te_raw, y_te)},
    "fp32_T":   {"ece": ece(p32_te_T, y_te),  "nll": nll(p32_te_T, y_te),  "brier": brier(p32_te_T, y_te)},
    "int8_raw": {"ece": ece(p8_te_raw, y_te), "nll": nll(p8_te_raw, y_te), "brier": brier(p8_te_raw, y_te)},
    "int8_T":   {"ece": ece(p8_te_T, y_te),   "nll": nll(p8_te_T, y_te),   "brier": brier(p8_te_T, y_te)},
    "agree_int8_vs_fp32_argmax": float(np.mean(p8_te_raw.argmax(1) == p32_te_raw.argmax(1))),
}
tmp = DEST.with_suffix(".tmp")
with open(tmp, "w") as f:
    json.dump(res, f, indent=2); f.flush(); os.fsync(f.fileno())
os.replace(tmp, DEST)
print(json.dumps(res, indent=2))
print(f"saved {DEST}")
