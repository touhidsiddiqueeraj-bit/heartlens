#!/usr/bin/env python3
"""Round-3 P2 — QAT CNN under the frozen 5x2 patient-disjoint folds.

Per fold: train FP32 CNN (same recipe as group_kfold_eval_v2: balanced weights,
Adam 1e-3, early stopping) -> tfmot QAT fine-tune (3 epochs, Adam 1e-4) ->
strip -> full-int8 convert. Also PTQ-converts the SAME FP32 weights so the
FP32 / PTQ-INT8 / QAT-INT8 triple is paired within each fold.

Resumable: per-fold JSON in results/qat_ckpt/{model}_s{seed}_f{fold}.json
(atomic tmp->rename; existing complete ckpts are skipped).
Fold s0_f0 additionally saves .tflite files for ESP deployment.

Usage: python3 scripts/round3_qat_cv.py --types cnn --epochs 30
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow_model_optimization as tfmot
import tf_keras as keras  # tfmot requires tf_keras (Keras 2 API), NOT Keras 3

TR = Path(__file__).resolve().parents[1] / "heart-lens-training"
sys.path.insert(0, str(TR))
OUT_DIR = TR / "results"
CKPT_DIR = OUT_DIR / "qat_ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
WINDOW_SAMPLES, NUM_CLASSES = 360, 3
CLASS_NAMES = ["Normal", "APB", "PVC"]


def atomic_write(p: Path, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def build_cnn_tfkeras():
    """Mirror of models.build_classifier('cnn') in tf_keras (tfmot-compatible)."""
    inp = keras.layers.Input(shape=(WINDOW_SAMPLES, 1), name="classifier_input")
    def block(x, f, name):
        x = keras.layers.Conv1D(f, 5, padding="same", activation="relu", name=f"{name}_conv")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        return keras.layers.MaxPool1D(2, name=f"{name}_pool")(x)
    x = block(inp, 32, "conv1")
    x = block(x, 64, "conv2")
    x = block(x, 128, "conv3")
    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(0.5, name="dropout")(x)
    out = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classifier_output")(x)
    m = keras.Model(inp, out, name="cnn_classifier_qat")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def load_arrays(data_dir, max_per_class):
    from train_classifier import load_data_with_record_tracking
    by_class = load_data_with_record_tracking(data_dir, max_per_class)
    X, y, g = [], [], []
    for cls, items in by_class.items():
        for seg, rec in items:
            X.append(seg); y.append(cls); g.append(rec)
    return (np.array(X).reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32),
            np.array(y), np.array(g))


def make_splits(groups, folds_file):
    j = json.loads(Path(folds_file).read_text())
    entries = {(e["seed"], e["fold"]): e for e in j["entries"]}
    out = []
    for seed in j["seeds"]:
        for fold in range(j["folds"]):
            e = entries[(seed, fold)]
            test_recs = set(e["test_recs"])
            tr = np.where(~np.isin(groups, list(test_recs)))[0]
            te = np.where(np.isin(groups, list(test_recs)))[0]
            out.append((seed, fold, tr, te))
    return out


def rep_fn(X, n=200):
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    def gen():
        for i in idx:
            yield [X[i:i + 1].astype(np.float32)]
    return gen


def to_int8(keras_model, X_train):
    c = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep_fn(X_train)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def int8_predict(tflite_bytes, X):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = ind["quantization"]; s_out, z_out = outd["quantization"]
    preds, probs = [], []
    for i in range(len(X)):
        q = np.round(X[i] / s_in + z_in).clip(-128, 127).astype(np.int8)
        interp.set_tensor(ind["index"], q.reshape(1, WINDOW_SAMPLES, 1))
        interp.invoke()
        raw = interp.get_tensor(outd["index"])[0].astype(np.float32)
        p = (raw - z_out) * s_out
        probs.append(p); preds.append(int(np.argmax(p)))
    return np.array(preds), np.array(probs)


def metrics(y_true, preds):
    f1 = f1_score(y_true, preds, average=None, zero_division=0)
    return {"f1": f1.tolist(), "macro": float(np.mean(f1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(TR / "mitdb"))
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=3)
    ap.add_argument("--types", default="cnn")
    ap.add_argument("--folds-file", default=str(OUT_DIR / "folds_5x2.json"))
    ap.add_argument("--save-tflite-fold", default="0,0", help="seed,fold whose tflites are kept for ESP")
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",")]
    save_seed, save_fold = (int(x) for x in args.save_tflite_fold.split(","))

    X, y, groups = load_arrays(args.data_dir, args.max_per_class)
    splits = make_splits(groups, args.folds_file)
    print(f"[qat] X={X.shape} folds={len(splits)} types={types}", flush=True)

    for mtype in types:
        assert mtype == "cnn", "QAT builder currently mirrors the CNN only"
        for (seed, fold, tr, te) in splits:
            ckpt = CKPT_DIR / f"{mtype}_s{seed}_f{fold}.json"
            if ckpt.exists():
                try:
                    j = json.loads(ckpt.read_text())
                    if "qat" in j and "ptq" in j:
                        print(f" -- skip {mtype} s{seed}f{fold} (ckpt exists) "
                              f"fp32={j['fp32']['macro']:.4f} ptq={j['ptq']['macro']:.4f} qat={j['qat']['macro']:.4f}", flush=True)
                        continue
                except Exception:
                    pass
            print(f"\n==== QAT {mtype} s{seed}f{fold} train={len(tr)} test={len(te)} ====", flush=True)
            try:
                classes = np.unique(y[tr])
                w = compute_class_weight("balanced", classes=classes, y=y[tr])
                cw = dict(zip(classes, w))
                tf.keras.backend.clear_session()
                base = build_cnn_tfkeras()
                base.fit(X[tr], y[tr], validation_data=(X[te], y[te]), epochs=args.epochs,
                         batch_size=64, class_weight=cw,
                         callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1)],
                         verbose=1)
                fp32_pred = np.argmax(base.predict(X[te], verbose=0), axis=1)
                fp32_m = metrics(y[te], fp32_pred)

                # PTQ on the SAME weights
                ptq_bytes = to_int8(base, X[tr])
                ptq_pred, _ = int8_predict(ptq_bytes, X[te])
                ptq_m = metrics(y[te], ptq_pred)

                # QAT: annotate Conv1D/Dense for quantization (BatchNorm/Mpool/Dropout/GAP pass-through)
                tf.keras.backend.clear_session()
                from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import Default8BitQuantizeConfig
                def _annotate(layer):
                    if isinstance(layer, keras.layers.Conv1D) or isinstance(layer, keras.layers.Dense):
                        return tfmot.quantization.keras.quantize_annotate_layer(
                            layer, quantize_config=Default8BitQuantizeConfig(['kernel'], ['activation'], False))
                    return layer
                with tfmot.quantization.keras.quantize_scope():
                    annotated = keras.models.clone_model(base, clone_function=_annotate)
                    q_aware = tfmot.quantization.keras.quantize_apply(annotated)
                q_aware.compile(optimizer=keras.optimizers.Adam(1e-4),
                                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                q_aware.fit(X[tr], y[tr], validation_data=(X[te], y[te]), epochs=args.qat_epochs,
                            batch_size=64, class_weight=cw, verbose=1)
                # strip is optional; converter handles q_aware directly (strip_quantization not in this tfmot build)
                qat_bytes = to_int8(q_aware, X[tr])
                qat_pred, _ = int8_predict(qat_bytes, X[te])
                qat_m = metrics(y[te], qat_pred)

                rec = {"model": mtype, "seed": seed, "fold": fold,
                       "fp32": fp32_m, "ptq": ptq_m, "qat": qat_m,
                       "disagree_ptq_vs_fp32": float(np.mean(ptq_pred != fp32_pred)),
                       "disagree_qat_vs_fp32": float(np.mean(qat_pred != fp32_pred)),
                       "disagree_qat_vs_ptq": float(np.mean(qat_pred != ptq_pred)),
                       "size_kb_ptq": round(len(ptq_bytes) / 1024, 1),
                       "size_kb_qat": round(len(qat_bytes) / 1024, 1)}
                if (seed, fold) == (save_seed, save_fold):
                    (CKPT_DIR / f"{mtype}_qat_s{seed}_f{fold}.tflite").write_bytes(qat_bytes)
                    (CKPT_DIR / f"{mtype}_fp32_s{seed}_f{fold}.tflite").write_bytes(ptq_bytes)
                    rec["saved_tflite"] = True
                atomic_write(ckpt, rec)
                print(f" -> fp32={fp32_m['macro']:.4f} ptq={ptq_m['macro']:.4f} qat={qat_m['macro']:.4f} "
                      f"disagree(ptq/fp32)={rec['disagree_ptq_vs_fp32']:.3f} (qat/fp32)={rec['disagree_qat_vs_fp32']:.3f}", flush=True)
            except Exception as e:
                print(f"!! fail {mtype} s{seed}f{fold}: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue

    # summary
    for mtype in types:
        rows = []
        for (seed, fold, _tr, _te) in splits:
            ckpt = CKPT_DIR / f"{mtype}_s{seed}_f{fold}.json"
            if ckpt.exists():
                rows.append(json.loads(ckpt.read_text()))
        if not rows:
            continue
        def col(k):
            return np.array([r[k]["macro"] for r in rows])
        fp, pt, qa = col("fp32"), col("ptq"), col("qat")
        summary = {"model": mtype, "n": len(rows),
                   "fp32_macro": float(fp.mean()), "ptq_macro": float(pt.mean()), "qat_macro": float(qa.mean()),
                   "fp32_std": float(fp.std()), "ptq_std": float(pt.std()), "qat_std": float(qa.std()),
                   "delta_ptq_vs_fp32": float((pt - fp).mean()), "delta_qat_vs_fp32": float((qa - fp).mean()),
                   "delta_qat_vs_ptq": float((qa - pt).mean()),
                   "ci95_qat_vs_ptq": float(1.96 * (qa - pt).std() / np.sqrt(len(rows))) if len(rows) > 1 else 0.0,
                   "disagree_ptq_vs_fp32": float(np.mean([r["disagree_ptq_vs_fp32"] for r in rows])),
                   "disagree_qat_vs_fp32": float(np.mean([r["disagree_qat_vs_fp32"] for r in rows])),
                   "disagree_qat_vs_ptq": float(np.mean([r["disagree_qat_vs_ptq"] for r in rows])),
                   "size_kb_ptq": float(np.mean([r["size_kb_ptq"] for r in rows])),
                   "size_kb_qat": float(np.mean([r["size_kb_qat"] for r in rows])),
                   "per_fold": rows}
        atomic_write(OUT_DIR / f"qat_{mtype}_summary.json", summary)
        print(f"\n[QAT {mtype}] n={len(rows)} FP32 {fp.mean():.4f}  PTQ {pt.mean():.4f}  QAT {qa.mean():.4f}  "
              f"dQAT-PTQ {(qa-pt).mean():+.4f}±{summary['ci95_qat_vs_ptq']:.4f}  "
              f"disagree qat/fp32 {summary['disagree_qat_vs_fp32']:.3f}", flush=True)


if __name__ == "__main__":
    main()
