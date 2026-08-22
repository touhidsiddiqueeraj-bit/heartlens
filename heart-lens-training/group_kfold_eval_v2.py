#!/usr/bin/env python3
"""Experiment 1 — Patient-independent grouped CV, 4-arch, per-fold resume.

Reboot-safe: each (seed,fold,model) writes results/group_kfold_ckpt/{model}_s{seed}_f{fold}.json atomically.
Merged summary -> results/group_kfold_all.json + per-model group_kfold_{model}.json.

Foreground: verbose=1, unbuffered prints.

Usage:
  python3 group_kfold_eval_v2.py --types cnn,tcn --folds 5 --seeds 0,1 --epochs 30
  python3 group_kfold_eval_v2.py --types cnn,lstm,gru,tcn --folds 5 --seeds 0,1 --epochs 30 --folds-file results/folds_5x2.json
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from data_loader import WINDOW_SAMPLES, NUM_CLASSES
from models import build_classifier

OUT_DIR = Path(__file__).parent / "results"
CKPT_DIR = OUT_DIR / "group_kfold_ckpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal","APB","PVC"]

def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def load_arrays(data_dir, max_per_class):
    from train_classifier import load_data_with_record_tracking
    by_class = load_data_with_record_tracking(data_dir, max_per_class)
    X_all, y_all, g_all = [], [], []
    for cls, items in by_class.items():
        for seg, rec in items:
            X_all.append(seg); y_all.append(cls); g_all.append(rec)
    X = np.array(X_all).reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    return X, np.array(y_all), np.array(g_all)

def run_fold(X, y, train_idx, test_idx, mtype, epochs):
    # weighted for APB/PVC vs Normal — consistent across folds (ponytail: simple class_weight)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y[train_idx])
    try:
        w = compute_class_weight("balanced", classes=classes, y=y[train_idx])
        cw = dict(zip(classes, w))
    except Exception:
        cw = None
    model = build_classifier(model_type=mtype)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
    ]
    model.fit(X[train_idx], y[train_idx],
              validation_data=(X[test_idx], y[test_idx]),
              epochs=epochs, batch_size=64,
              class_weight=cw,
              callbacks=callbacks, verbose=1)
    probs = model.predict(X[test_idx], verbose=0)
    pred = np.argmax(probs, axis=1)
    f1 = f1_score(y[test_idx], pred, average=None, zero_division=0)
    # per-class precision/recall for APB analysis
    prec, rec, _, support = precision_recall_fscore_support(y[test_idx], pred, labels=[0,1,2], zero_division=0)
    return {"f1": f1.tolist(), "prec": prec.tolist(), "rec": rec.tolist(),
            "support": support.tolist(), "macro": float(np.mean(f1))}, pred.tolist()

def make_splits(X, y, groups, folds, seeds, folds_file=None):
    # preferred: use frozen folds_5x2.json directly (reboot-safe, identical for every arch)
    if folds_file and Path(folds_file).exists():
        j = json.loads(Path(folds_file).read_text())
        print(f"[splits] using frozen {folds_file} (folds={j['folds']} seeds={j['seeds']})")
        entries = {(e["seed"], e["fold"]): e for e in j["entries"]}
        out = []
        for seed in seeds:
            for fold in range(folds):
                e = entries.get((seed, fold))
                if e is None:
                    continue
                test_recs = set(e["test_recs"])
                tr = np.where(~np.isin(groups, list(test_recs)))[0]
                te = np.where(np.isin(groups, list(test_recs)))[0]
                out.append((seed, fold, tr, te))
        return out
    # fallback: chunk permuted uniq records (no GroupKFold needed)
    out=[]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        uniq = np.array(sorted(set(groups.tolist())))
        perm = rng.permutation(uniq)
        fold_groups = np.array_split(perm, folds)
        for fold, test_recs_arr in enumerate(fold_groups):
            test_recs = set(test_recs_arr.tolist())
            tr = np.where(~np.isin(groups, list(test_recs)))[0]
            te = np.where(np.isin(groups, list(test_recs)))[0]
            out.append((seed,fold,tr,te))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./mitdb")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--types", type=str, default="cnn,tcn",
                    help="comma-separated cnn,lstm,gru,tcn (ponytail: default cnn,tcn for 12h)")
    ap.add_argument("--folds-file", type=str, default="results/folds_5x2.json")
    ap.add_argument("--balanced", action="store_true", help="force balanced class_weight even if not needed")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    folds = args.folds

    X, y, groups = load_arrays(args.data_dir, args.max_per_class)
    print(f"[group_kfold_v2] X={X.shape} classes={np.bincount(y)} records={len(set(groups))} types={types}", flush=True)
    print(f"[group_kfold_v2] folds={folds} seeds={seeds} epochs={args.epochs}", flush=True)
    print(f"[group_kfold_v2] ckpts in {CKPT_DIR} (per-fold resume)", flush=True)

    splits = make_splits(X, y, groups, folds, seeds, args.folds_file if Path(args.folds_file).exists() else None)

    total = len(types) * len(splits)
    done_ckpts = sum(1 for m in types for (s,f,_,_) in splits if (CKPT_DIR / f"{m}_s{s}_f{f}.json").exists())
    def pct_str(done, total):
        p = (done/total*100) if total else 100
        bar_len = 20
        filled = int(bar_len * done / total) if total else bar_len
        bar = "█"*filled + "░"*(bar_len - filled)
        return f"{p:5.1f}% [{bar}] {done}/{total}"

    print(f"[progress] {pct_str(done_ckpts, total)} overall  ({done_ckpts}/{total} folds ckpt'd)", flush=True)

    # progress helper — keep-here file so you can check anytime
    PROG_FILE = OUT_DIR / "progress.txt"
    def write_progress(done, total, cur_mtype=None, cur_seed=None, cur_fold=None, status=""):
        p = (done/total*100) if total else 100
        with open(PROG_FILE, "w") as pf:
            pf.write(f"PROGRESS {p:.1f}% ({done}/{total}) {cur_mtype or ''} s{cur_seed if cur_seed is not None else ''}f{cur_fold if cur_fold is not None else ''} {status}\n")
            pf.write(f"bar: {pct_str(done,total)}\n")

    write_progress(done_ckpts, total, status="running")

    # resume: skip existing ckpts
    cur_done = done_ckpts
    for mtype in types:
        for (seed, fold, tr, te) in splits:
            ckpt = CKPT_DIR / f"{mtype}_s{seed}_f{fold}.json"
            if ckpt.exists():
                try:
                    j = json.loads(ckpt.read_text())
                    if "f1" in j and len(j["f1"])==3:
                        print(f"  -- skip {mtype} s{seed}f{fold} (ckpt exists) F1={np.round(j['f1'],4)} macro={j.get('macro',0):.4f}  {pct_str(cur_done, total)}", flush=True)
                        continue
                except Exception:
                    pass
            # starting new fold — update progress file
            write_progress(cur_done, total, mtype, seed, fold, "training")
            print(f"\n==== {mtype.upper()} s{seed} fold{fold}  train={len(tr)} test={len(te)} test_counts={np.bincount(y[te], minlength=3).tolist()}  {pct_str(cur_done, total)} overall ====", flush=True)
            try:
                res, pred = run_fold(X, y, tr, te, mtype, args.epochs)
            except Exception as e:
                print(f"!! fail {mtype} s{seed}f{fold}: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue
            out = {"model": mtype, "seed": seed, "fold": fold,
                   "f1": res["f1"], "prec": res["prec"], "rec": res["rec"],
                   "support": res["support"], "macro": res["macro"]}
            atomic_write_json(ckpt, out)
            cur_done += 1
            write_progress(cur_done, total, mtype, seed, fold, f"saved macro={res['macro']:.4f}")
            print(f"  -> saved {ckpt}  F1={np.round(res['f1'],4)} macro={res['macro']:.4f}  {pct_str(cur_done, total)}", flush=True)

    # merge summaries
    merged = {mtype: [] for mtype in types}
    for mtype in types:
        for (seed, fold, _tr, _te) in splits:
            ckpt = CKPT_DIR / f"{mtype}_s{seed}_f{fold}.json"
            if ckpt.exists():
                j = json.loads(ckpt.read_text())
                merged[mtype].append(j)

    for mtype in types:
        rows = merged[mtype]
        if not rows:
            print(f"[merge] no ckpts for {mtype}", flush=True)
            continue
        arr = np.array([r["f1"] for r in rows])
        mean = arr.mean(axis=0); std = arr.std(axis=0)
        macro_mean = float(np.mean(arr.mean(axis=1))); macro_std = float(arr.mean(axis=1).std())
        # 95% CI via normal approx
        n = len(rows)
        ci95 = 1.96 * np.array([arr[:,i].std()/np.sqrt(n) for i in range(3)])
        macro_ci = 1.96 * macro_std/np.sqrt(n) if n>1 else 0
        summary = {"model": mtype, "class_names": CLASS_NAMES,
                   "n": n, "folds": folds, "seeds": seeds,
                   "mean": mean.tolist(), "std": std.tolist(),
                   "ci95": ci95.tolist(),
                   "macro_mean": macro_mean, "macro_std": macro_std, "macro_ci95": macro_ci,
                   "per_fold": rows}
        # per-model file
        atomic_write_json(OUT_DIR / f"group_kfold_{mtype}.json", summary)
        print(f"\n[{mtype}] n={n}  Normal {mean[0]:.4f}±{std[0]:.4f}  APB {mean[1]:.4f}±{std[1]:.4f}  PVC {mean[2]:.4f}±{std[2]:.4f}  Macro {macro_mean:.4f}±{macro_std:.4f} ci95={macro_ci:.4f}", flush=True)

    # combined file for paper
    atomic_write_json(OUT_DIR / "group_kfold_all.json", {k: {"mean": np.array([r["f1"] for r in v]).mean(axis=0).tolist() if v else [],
                                                           "rows": v} for k,v in merged.items()})
    print(f"\nSaved {OUT_DIR/'group_kfold_all.json'} and per-model group_kfold_{{model}}.json", flush=True)
    # also write legacy compat file for old paper table: use best deployable (tcn if present else cnn)
    best = "tcn" if "tcn" in merged and merged["tcn"] else ("cnn" if "cnn" in merged else types[0])
    if merged.get(best):
        arr = np.array([r["f1"] for r in merged[best]])
        legacy = {"class_names": CLASS_NAMES, "mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist(),
                  "macro_mean": float(arr.mean()), "macro_std": float(arr.std()), "folds": folds, "seeds": seeds}
        atomic_write_json(OUT_DIR / "group_kfold.json", legacy)
        print(f"Legacy {OUT_DIR/'group_kfold.json'} written from {best}", flush=True)

if __name__ == "__main__":
    main()
