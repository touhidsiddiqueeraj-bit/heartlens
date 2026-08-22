#!/usr/bin/env python3
"""APB imbalance ablation — scoped to CNN/TCN, single record split, 4 strategies.

Foreground + resume: each (model,strategy) is a checkpoint file.
Strategies:
  baseline  — no class_weight
  weighted  — sklearn balanced
  focal     — focal loss gamma=2 (alpha balanced)
  balanced  — balanced batch sampling via class_weight + oversample minority in-place (simple repeat)

Reuses record_level_split seed=42 (same as compare_models.py) so test set has 24 APB windows.

Usage:
  python3 scripts/apb_ablation.py --types cnn,tcn --epochs 30
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

TR = Path(__file__).resolve().parents[1] / "heart-lens-training"
sys.path.insert(0, str(TR))
OUT_DIR = TR / "results"
CKPT_DIR = OUT_DIR / "apb_ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal","APB","PVC"]

def atomic_write(p: Path, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def focal_loss(gamma=2., alpha=None):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_one = tf.one_hot(y_true, depth=3)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)  # not used directly
        # focal on softmax probs
        p_t = tf.reduce_sum(y_one * y_pred, axis=1)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0)
        ce_f = -tf.math.log(p_t)
        mod = tf.pow(1.0 - p_t, gamma)
        if alpha is not None:
            a_t = tf.gather(alpha, y_true)
            mod = mod * a_t
        return tf.reduce_mean(mod * ce_f)
    return loss

def build_with_strategy(mtype, strategy):
    from models import build_classifier
    model = build_classifier(model_type=mtype)
    if strategy == "focal":
        # alpha = balanced weights normalized
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=focal_loss(gamma=2., alpha=None),
                      metrics=["accuracy"])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="cnn,tcn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--strategies", default="baseline,weighted,focal,balanced")
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    from train_classifier import load_data_with_record_tracking, record_level_split
    from data_loader import WINDOW_SAMPLES
    by_class = load_data_with_record_tracking(str(TR/"mitdb"), max_per_class=args.max_per_class)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = record_level_split(by_class, train_ratio=0.7, val_ratio=0.15, seed=42)
    for arr in [X_tr, X_va, X_te]:
        pass
    X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_va = X_va.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_te = X_te.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    total = len(types)*len(strategies)
    done0 = sum(1 for m in types for s in strategies if (CKPT_DIR/f"{m}_{s}.json").exists())
    print(f"[apb] Train {X_tr.shape} {np.bincount(y_tr)}  Val {X_va.shape}  Test {X_te.shape} {np.bincount(y_te)}  {done0}/{total} {done0/total*100:.1f}% done", flush=True)
    APB_PROG = OUT_DIR/"apb_progress.txt"
    def apb_write(done, total, cur=""):
        with open(APB_PROG,"w") as f: f.write(f"APB {done}/{total} {done/total*100:.1f}% {cur}\n")
    apb_write(done0, total)
    cur_done = done0
    for mtype in types:
        for strat in strategies:
            ckpt = CKPT_DIR / f"{mtype}_{strat}.json"
            if ckpt.exists():
                try:
                    j = json.loads(ckpt.read_text())
                    if "f1" in j:
                        print(f" -- skip {mtype}/{strat} (ckpt exists) F1={j['f1']}  {cur_done}/{total} {cur_done/total*100:.1f}%", flush=True)
                        continue
                except Exception:
                    pass
            apb_write(cur_done, total, f"{mtype}/{strat} training")
            print(f"\n==== {mtype.upper()} strat={strat} [{cur_done}/{total} {cur_done/total*100:.1f}%] ====", flush=True)
            from models import build_classifier
            model = build_classifier(model_type=mtype)
            # select loss/cw per strategy
            cw = None
            loss = "sparse_categorical_crossentropy"
            if strat == "weighted":
                classes = np.unique(y_tr)
                w = compute_class_weight("balanced", classes=classes, y=y_tr)
                cw = dict(zip(classes, w))
                print(f"  class_weight={cw}", flush=True)
            elif strat == "focal":
                classes = np.unique(y_tr)
                w = compute_class_weight("balanced", classes=classes, y=y_tr)
                alpha = np.array([w[np.where(classes==c)[0][0]] if c in classes else 1.0 for c in range(3)], dtype=np.float32)
                alpha = alpha / alpha.mean()
                loss = focal_loss(gamma=2., alpha=alpha)
                print(f"  focal gamma=2 alpha={alpha}", flush=True)
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
                cw = None
            elif strat == "balanced":
                # oversample minority by repeating APB/PVC to match Normal count (ponytail: simple repeat, no SMOTE)
                max_c = max(np.bincount(y_tr))
                Xb, yb = [], []
                for c in range(3):
                    mask = y_tr==c
                    Xc = X_tr[mask]; yc = y_tr[mask]
                    if len(yc)==0: continue
                    rep = max_c // len(yc)
                    rem = max_c % len(yc)
                    Xc_rep = np.concatenate([Xc]*rep + ([Xc[:rem]] if rem else []))
                    yc_rep = np.concatenate([yc]*rep + ([yc[:rem]] if rem else []))
                    Xb.append(Xc_rep); yb.append(yc_rep)
                X_tr_b = np.concatenate(Xb); y_tr_b = np.concatenate(yb)
                # shuffle
                idx = np.random.permutation(len(y_tr_b))
                X_tr_use, y_tr_use = X_tr_b[idx], y_tr_b[idx]
                print(f"  balanced oversample: {np.bincount(y_tr)} -> {np.bincount(y_tr_use)}", flush=True)
            else:
                X_tr_use, y_tr_use = X_tr, y_tr

            if strat != "balanced":
                X_tr_use, y_tr_use = X_tr, y_tr
            if strat == "focal":
                # already compiled
                callbacks = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
                             tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)]
                model.fit(X_tr_use, y_tr_use, validation_data=(X_va, y_va), epochs=args.epochs, batch_size=64, callbacks=callbacks, verbose=1)
            else:
                if strat not in ("focal",):
                    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
                callbacks = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
                             tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)]
                model.fit(X_tr_use, y_tr_use, validation_data=(X_va, y_va), epochs=args.epochs, batch_size=64, class_weight=cw, callbacks=callbacks, verbose=1)

            pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
            f1 = f1_score(y_te, pred, average=None, zero_division=0)
            prec, rec, _, sup = precision_recall_fscore_support(y_te, pred, labels=[0,1,2], zero_division=0)
            per = {"model": mtype, "strategy": strat, "f1": f1.tolist(), "prec": prec.tolist(), "rec": rec.tolist(),
                   "support": sup.tolist(), "macro": float(np.mean(f1)),
                   "apb_f1": float(f1[1]), "apb_prec": float(prec[1]), "apb_rec": float(rec[1])}
            atomic_write(ckpt, per)
            cur_done += 1
            apb_write(cur_done, total, f"{mtype}/{strat} saved")
            print(f"  -> {ckpt}  F1={np.round(f1,4)} macro={per['macro']:.4f} APB F1={per['apb_f1']:.4f} (P={per['apb_prec']:.4f} R={per['apb_rec']:.4f})  [{cur_done}/{total} {cur_done/total*100:.1f}%]", flush=True)

    # merge summary table (paper Table III)
    rows=[]
    for mtype in types:
        for strat in strategies:
            ckpt = CKPT_DIR/f"{mtype}_{strat}.json"
            if ckpt.exists():
                rows.append(json.loads(ckpt.read_text()))
    atomic_write(OUT_DIR/"apb_ablation.json", rows)
    print(f"\nSaved {OUT_DIR/'apb_ablation.json'} rows={len(rows)}", flush=True)
    # pretty table
    print("\n=== APB ablation summary ===", flush=True)
    for r in rows:
        print(f"  {r['model']:4s} {r['strategy']:10s} macro={r['macro']:.4f} APB F1={r['apb_f1']:.4f} P={r['apb_prec']:.4f} R={r['apb_rec']:.4f}", flush=True)

if __name__ == "__main__":
    main()
