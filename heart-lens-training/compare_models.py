#!/usr/bin/env python3
"""Model comparison — CNN vs LSTM vs GRU vs TCN on the same pipeline.

Paper contribution (review #18): a comparative evaluation of deep
learning architectures for abnormal heart-rhythm detection on edge
devices. Reports per-class F1, int8 TFLite size, and quantization
delta for each architecture. Latency is measured on hardware via the
firmware BENCHMARK_MODE (fill into hw_eval/latency.csv).

Usage:
    python3 compare_models.py --epochs 40
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from data_loader import WINDOW_SAMPLES, NUM_CLASSES
from models import build_classifier

OUT_DIR = Path(__file__).parent / "results"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "APB", "PVC"]
MODEL_TYPES = ["cnn", "lstm", "gru", "tcn"]


def quantize(model, X_val):
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
    return converter.convert()


def eval_quantized(tflite_model, X_test, y_test):
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_s, in_z = in_d["quantization"]
    out_s, out_z = out_d["quantization"]
    preds = []
    for i in range(len(X_test)):
        x = np.round(X_test[i] / in_s + in_z).clip(-128, 127).astype(np.int8)
        interp.set_tensor(in_d["index"], x.reshape(1, WINDOW_SAMPLES, 1))
        interp.invoke()
        raw = interp.get_tensor(out_d["index"])[0]
        probs = (raw.astype(np.float32) - out_z) * out_s
        preds.append(np.argmax(probs))
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./mitdb")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models-dir", default="./models")
    ap.add_argument("--types", default=",".join(MODEL_TYPES),
                    help="comma-separated architectures to compare")
    args = ap.parse_args()

    from train_classifier import load_data_with_record_tracking, record_level_split
    by_class = load_data_with_record_tracking(args.data_dir, args.max_per_class)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = record_level_split(by_class)
    X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_va = X_va.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_te = X_te.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    print(f"Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")

    os.makedirs(args.models_dir, exist_ok=True)
    rows = []
    for mtype in args.types.split(","):
        mtype = mtype.strip()
        if mtype not in MODEL_TYPES:
            print(f"skipping unknown model type: {mtype}")
            continue
        print(f"\n=== {mtype.upper()} ===")
        keras_path = os.path.join(args.models_dir, f"compare_{mtype}.keras")
        if os.path.exists(keras_path):
            model = tf.keras.models.load_model(keras_path)
        else:
            model = build_classifier(model_type=mtype)
            model.fit(X_tr, y_tr, epochs=args.epochs, batch_size=64,
                      validation_data=(X_va, y_va),
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,
                                                                  restore_best_weights=True)],
                      verbose=0)
            model.save(keras_path)

        pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        f1_float = f1_score(y_te, pred, average=None, zero_division=0)
        print(f"float32 per-class F1: {np.round(f1_float, 4)}  "
              f"macro={np.mean(f1_float):.4f}")

        tflite_path = os.path.join(args.models_dir, f"compare_{mtype}_int8.tflite")
        if os.path.exists(tflite_path):
            tflite_model = open(tflite_path, "rb").read()
        else:
            tflite_model = quantize(model, X_va)
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

        pred_q = eval_quantized(tflite_model, X_te, y_te)
        f1_quant = f1_score(y_te, pred_q, average=None, zero_division=0)
        delta = float(np.mean(f1_float) - np.mean(f1_quant))
        print(f"int8 per-class F1:     {np.round(f1_quant, 4)}  "
              f"macro={np.mean(f1_quant):.4f}  (delta {delta:+.4f})")
        print(f"model size: {len(tflite_model)/1024:.1f} KB")

        rows.append({
            "model": mtype,
            "size_kb": round(len(tflite_model) / 1024, 1),
            "macro_f1_float32": round(float(np.mean(f1_float)), 4),
            "macro_f1_int8": round(float(np.mean(f1_quant)), 4),
            "quant_delta": round(delta, 4),
            "f1_normal": round(float(f1_float[0]), 4),
            "f1_apb": round(float(f1_float[1]), 4),
            "f1_pvc": round(float(f1_float[2]), 4),
            "latency_ms": None,  # measured on hardware via BENCHMARK_MODE
        })

    csv_path = OUT_DIR / "model_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_DIR / "model_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["model"].upper() for r in rows]
    f1s = [r["macro_f1_float32"] for r in rows]
    sizes = [r["size_kb"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(names, f1s, color="#4a7ba6")
    a1.set_ylabel("Macro F1 (float32)")
    a1.set_ylim(0, 1)
    a1.set_title("Accuracy vs architecture")
    a2.bar(names, sizes, color="#b0604a")
    a2.set_ylabel("int8 model size (KB)")
    a2.set_title("Size vs architecture")
    for ax in (a1, a2):
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / "model_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
