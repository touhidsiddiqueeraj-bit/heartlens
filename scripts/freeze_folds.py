#!/usr/bin/env python3
"""Freeze record-level fold IDs so every arch uses identical patient splits.

Writes heart-lens-training/results/folds_5x2.json with:
  {seeds: [0,1], folds: 5, items: [{seed, fold, train_recs, test_recs, test_counts}]}
Reboot-safe: atomic write via .tmp -> replace. Reuses existing file if present.
If seeds differ, file is regenerated.
"""
import json
import os
import sys
from pathlib import Path

TR = Path(__file__).resolve().parents[1] / "heart-lens-training"
sys.path.insert(0, str(TR))
OUT = TR / "results" / "folds_5x2.json"
TMP = OUT.with_suffix(".tmp")

FOLDS = 5
SEEDS = [0, 1]

def main():
    import numpy as np
    from sklearn.model_selection import GroupKFold
    sys.path.insert(0, str(TR))
    from train_classifier import load_data_with_record_tracking
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    tmp = out.with_suffix(".tmp")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    folds = args.folds

    # Reuse if already exists with same seeds/folds
    if out.exists():
        try:
            j = json.loads(out.read_text())
            if j.get("folds")==folds and j.get("seeds")==seeds:
                print(f"[freeze_folds] reuse {out} (folds={folds} seeds={seeds})")
                return
            else:
                print(f"[freeze_folds] config changed, regenerating {out}")
        except Exception:
            pass

    print(f"[freeze_folds] loading data (max_per_class={args.max_per_class}) ...")
    by_class = load_data_with_record_tracking(str(TR/"mitdb"), max_per_class=args.max_per_class)
    # Build X,y,groups arrays like group_kfold_eval
    from data_loader import WINDOW_SAMPLES
    X_all, y_all, g_all = [], [], []
    for cls, items in by_class.items():
        for seg, rec in items:
            X_all.append(seg); y_all.append(cls); g_all.append(rec)
    X = np.array(X_all).reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    y = np.array(y_all); groups = np.array(g_all)
    print(f"  X={X.shape} classes={np.bincount(y)} records={len(set(groups))}")

    entries = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        uniq = np.array(sorted(set(groups.tolist())))
        perm = rng.permutation(uniq)
        # chunk perm into folds
        fold_groups = np.array_split(perm, folds)
        for fold, test_recs_arr in enumerate(fold_groups):
            test_recs = sorted(test_recs_arr.tolist())
            train_recs = sorted([r for r in uniq if r not in set(test_recs)])
            te_mask = np.isin(groups, test_recs)
            bc = np.bincount(y[te_mask], minlength=3).tolist()
            entries.append({"seed": seed, "fold": fold,
                            "train_recs": train_recs, "test_recs": test_recs,
                            "test_counts": bc})
            print(f"  seed={seed} fold={fold} test_recs={len(test_recs)} counts={bc} test={test_recs[:3]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"folds": folds, "seeds": seeds, "entries": entries,
               "records": sorted(set(groups.tolist()))}
    # atomic write
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    print(f"[freeze_folds] saved {out}")

if __name__ == "__main__":
    main()
