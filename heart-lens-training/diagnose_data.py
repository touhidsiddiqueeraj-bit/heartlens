#!/usr/bin/env python3
"""Diagnose data composition for the split/eval mismatch (no training).

Prints per-class record coverage after capping, per-split y bincounts,
NaN/range stats, and the GroupKFold fold-0 baseline for contrast.
Run:  python3 diagnose_data.py
"""

import argparse

import numpy as np
from sklearn.model_selection import GroupKFold

from train_classifier import load_data_with_record_tracking, record_level_split
from group_kfold_eval import load_arrays
from data_loader import load_record_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./mitdb")
    ap.add_argument("--max-per-class", type=int, default=3000)
    args = ap.parse_args()

    by_class = load_data_with_record_tracking(args.data_dir, args.max_per_class)
    for c, items in by_class.items():
        recs = sorted(set(rec for _, rec in items))
        print(f"train_classifier by_class[{c}] n={len(items)} "
              f"records={len(recs)} {recs}")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = record_level_split(by_class)
    print("\nrecord_level_split y bincounts:")
    for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        print(f"  {name}: {np.bincount(y)}")
    for name, X in [("X_tr", X_tr), ("X_va", X_va), ("X_te", X_te)]:
        print(f"  {name}: shape={X.shape} nan={np.isnan(X).sum()} "
              f"min={X.min():.4f} max={X.max():.4f}")

    rec_segs = load_record_segments(args.data_dir, args.max_per_class)
    classes_per_rec = {}
    for rec, segs in rec_segs.items():
        for _, cls, _ in segs:
            classes_per_rec.setdefault(cls, set()).add(rec)
    print("\nload_record_segments (good loader) per-class record coverage:")
    for c, recs in sorted(classes_per_rec.items()):
        print(f"  class {c}: {len(recs)} records {sorted(recs)}")

    X, y, groups = load_arrays(args.data_dir, args.max_per_class)
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    perm = {r: i for i, r in enumerate(rng.permutation(uniq))}
    g_perm = np.array([perm[g] for g in groups])
    gkf = GroupKFold(n_splits=5)
    print("\nGroupKFold seed=0 test-fold composition (healthy baseline):")
    for fold, (_, te) in enumerate(gkf.split(X, y, groups=g_perm)):
        print(f"  fold {fold}: test records={sorted(set(groups[te]))} "
              f"y bincount={np.bincount(y[te])}")


if __name__ == "__main__":
    main()
