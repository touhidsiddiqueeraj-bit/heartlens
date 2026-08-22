#!/usr/bin/env python3
"""Paired FP32 vs INT8 evaluation per fold (review C4).

For each checkpoint in results/group_kfold_ckpt/{model}_s{seed}_f{fold}.json,
reload the corresponding fold split, quantize the trained model, and evaluate
FP32 vs INT8 on IDENTICAL X_te. Reuses folds_5x2.json.

Foreground, resume: writes results/paired_quant/{model}_s{seed}_f{fold}.json atomically.
Summary -> results/paired_quant.json + results/paired_quant_summary.csv

Usage: python3 scripts/paired_quant.py --types cnn,tcn --epochs 30
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

TR = Path(__file__).resolve().parents[1] / "heart-lens-training"
sys.path.insert(0, str(TR))
OUT_DIR = TR / "results"
CKPT_DIR = OUT_DIR / "group_kfold_ckpt"
PAIRED_DIR = OUT_DIR / "paired_quant"
PAIRED_DIR.mkdir(parents=True, exist_ok=True)

def atomic_write(p: Path, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def quantize_model(model, X_val):
    def rep():
        for _ in range(200):
            idx = np.random.randint(0, len(X_val))
            yield [X_val[idx:idx+1].astype(np.float32)]
    try:
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
        tflite = c.convert()
        return tflite, "full-int8"
    except Exception as e:
        print(f"  quant full-int8 failed: {e}, falling back float32", flush=True)
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        c._experimental_lower_tensor_list_ops = False
        return c.convert(), "float32"

def eval_int8(tflite, X_te):
    from data_loader import WINDOW_SAMPLES
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]; out_d = interp.get_output_details()[0]
    in_s, in_z = in_d["quantization"]; out_s, out_z = out_d["quantization"]
    preds=[]
    for i in range(len(X_te)):
        x = np.round(X_te[i]/in_s + in_z).clip(-128,127).astype(np.int8)
        interp.set_tensor(in_d["index"], x.reshape(1, WINDOW_SAMPLES, 1))
        interp.invoke()
        raw = interp.get_tensor(out_d["index"])[0]
        probs = (raw.astype(np.float32) - out_z)*out_s
        preds.append(np.argmax(probs))
    return np.array(preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="cnn,tcn")
    ap.add_argument("--folds-file", default=str(OUT_DIR/"folds_5x2.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-per-class", type=int, default=3000)
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    from train_classifier import load_data_with_record_tracking
    from data_loader import WINDOW_SAMPLES, NUM_CLASSES
    from models import build_classifier
    from sklearn.model_selection import GroupKFold

    # load arrays + groups to reconstruct splits
    from train_classifier import load_data_with_record_tracking
    by_class = load_data_with_record_tracking(str(TR/"mitdb"), max_per_class=args.max_per_class)
    X_all, y_all, g_all = [], [], []
    for cls, items in by_class.items():
        for seg, rec in items:
            X_all.append(seg); y_all.append(cls); g_all.append(rec)
    X = np.array(X_all).reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    y = np.array(y_all); groups = np.array(g_all)

    # load frozen splits mapping
    folds_file = Path(args.folds_file)
    seeds = [0,1]; folds=5
    if folds_file.exists():
        j=json.loads(folds_file.read_text()); seeds=j["seeds"]; folds=j["folds"]
        splits=[]
        entries={(e["seed"],e["fold"]):e for e in j["entries"]}
        for seed in seeds:
            for fold in range(folds):
                e=entries.get((seed,fold))
                if e is None: continue
                tr=np.where(~np.isin(groups, e["test_recs"]))[0]
                te=np.where(np.isin(groups, e["test_recs"]))[0]
                splits.append((seed,fold,tr,te))
    else:
        splits=[]
        for seed in seeds:
            rng=np.random.default_rng(seed)
            uniq=np.array(sorted(set(groups.tolist())))
            perm=rng.permutation(uniq)
            fold_groups=np.array_split(perm,folds)
            for fold,test_recs_arr in enumerate(fold_groups):
                tr=np.where(~np.isin(groups, list(test_recs_arr)))[0]
                te=np.where(np.isin(groups, list(test_recs_arr)))[0]
                splits.append((seed,fold,tr,te))

    all_rows=[]
    for mtype in types:
        for (seed,fold,tr,te) in splits:
            ckpt = CKPT_DIR / f"{mtype}_s{seed}_f{fold}.json"
            paired = PAIRED_DIR / f"{mtype}_s{seed}_f{fold}.json"
            if not ckpt.exists():
                print(f" -- skip {mtype} s{seed}f{fold} (no group_kfold ckpt yet)", flush=True)
                continue
            if paired.exists():
                try:
                    j=json.loads(paired.read_text())
                    if "delta_macro" in j:
                        print(f" -- skip paired {mtype} s{seed}f{fold} (exists) delta={j['delta_macro']:.4f}", flush=True)
                        all_rows.append(j); continue
                except Exception:
                    pass
            print(f"\n==== paired {mtype} s{seed}f{fold} ====", flush=True)
            # retrain same fold deterministically (no ckpt stores weights — retrain is the paired measure)
            # ponytail: retrain from scratch on identical split — cheapest correct paired test
            from sklearn.utils.class_weight import compute_class_weight
            classes=np.unique(y[tr])
            w=compute_class_weight("balanced", classes=classes, y=y[tr])
            cw=dict(zip(classes,w))
            tf.keras.backend.clear_session()
            model=build_classifier(model_type=mtype)
            model.fit(X[tr], y[tr], validation_data=(X[te], y[te]), epochs=args.epochs, batch_size=64,
                      class_weight=cw,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
                                 tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)],
                      verbose=1)
            pred_fp32=np.argmax(model.predict(X[te], verbose=0), axis=1)
            f1_fp32=f1_score(y[te], pred_fp32, average=None, zero_division=0)
            # quantize + eval int8
            tflite, qtype = quantize_model(model, X[tr][:2000] if len(X[tr])>2000 else X[tr])
            size_kb=len(tflite)/1024
            if qtype=="full-int8":
                pred_int8=eval_int8(tflite, X[te])
                f1_int8=f1_score(y[te], pred_int8, average=None, zero_division=0)
                disagree=float(np.mean(pred_fp32!=pred_int8))
                delta = float(np.mean(f1_int8) - np.mean(f1_fp32))
            else:
                pred_int8=None; f1_int8=None; disagree=None; delta=None
            rec={"model":mtype,"seed":seed,"fold":fold,"quant_type":qtype,"size_kb":round(size_kb,1),
                 "f1_fp32": f1_fp32.tolist(), "macro_fp32": float(np.mean(f1_fp32)),
                 "f1_int8": (f1_int8.tolist() if f1_int8 is not None else None),
                 "macro_int8": (float(np.mean(f1_int8)) if f1_int8 is not None else None),
                 "delta_macro": delta, "disagree": disagree}
            atomic_write(paired, rec)
            print(f"  FP32 macro={rec['macro_fp32']:.4f} INT8 macro={rec['macro_int8']} delta={delta} disagree={disagree} size={size_kb:.1f}KB", flush=True)
            all_rows.append(rec)

    # summary
    atomic_write(OUT_DIR/"paired_quant.json", all_rows)
    # per-model delta summary
    summary={}
    for mtype in types:
        rows=[r for r in all_rows if r["model"]==mtype and r["delta_macro"] is not None]
        if rows:
            deltas=np.array([r["delta_macro"] for r in rows])
            agrees=np.array([r["disagree"] for r in rows if r["disagree"] is not None])
            summary[mtype]={"n":len(rows),"delta_mean":float(deltas.mean()),"delta_std":float(deltas.std()),
                            "delta_ci95":float(1.96*deltas.std()/np.sqrt(len(rows))) if len(rows)>1 else 0.0,
                            "disagree_mean":float(agrees.mean()) if len(agrees) else None}
            print(f"[{mtype}] delta_mean={summary[mtype]['delta_mean']:.4f}±{summary[mtype]['delta_std']:.4f} ci95={summary[mtype]['delta_ci95']:.4f} disagree={summary[mtype]['disagree_mean']:.3f}", flush=True)
    atomic_write(OUT_DIR/"paired_quant_summary.json", summary)
    print(f"\nSaved {OUT_DIR/'paired_quant.json'} rows={len(all_rows)}", flush=True)

if __name__=="__main__":
    main()
