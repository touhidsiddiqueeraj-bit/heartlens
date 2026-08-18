#!/usr/bin/env python3
"""Experiment 6 — compute Delta-F1 = F1_digital - F1_hardware.

Reads digital.json (PC-side int8 TFLite predictions) and hardware.json
(board replay predictions), computes per-class and macro F1 for each leg,
and reports the hardware-domain shift Delta-F1.

Usage:
    python3 hw_eval/compute_delta.py \
        --digital hw_eval/captures/digital.json \
        --hardware hw_eval/captures/hardware.json
"""

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

CLASS_NAMES = ["Normal", "APB", "PVC"]


def f1s(labels, preds):
    return f1_score(labels, preds, average=None, zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digital", required=True)
    ap.add_argument("--hardware", required=True)
    args = ap.parse_args()

    with open(args.digital) as f:
        dig = json.load(f)
    with open(args.hardware) as f:
        hw = json.load(f)

    labels = np.array(dig["labels"])
    pred_d = np.array(dig["segments"])
    pred_h = np.array(hw["preds"])

    # Align: hardware leg may skip segments — keep intersection
    n_d = len(labels)
    n_h = len(pred_h)
    if n_h < n_d:
        print(f"WARNING: hardware leg has {n_h}/{n_d} results — "
              "using first n_h for both legs")
        labels, pred_d = labels[:n_h], pred_d[:n_h]

    fd = f1s(labels, pred_d)
    fh = f1s(labels, pred_h)
    delta = fd - fh

    print("\n=== Hardware-domain robustness (Delta-F1) ===")
    print(f"{'Class':8s} {'F1 digital':>10s} {'F1 hardware':>12s} {'Delta':>8s}")
    print("-" * 44)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:8s} {fd[i]:10.4f} {fh[i]:12.4f} {delta[i]:+8.4f}")
    print(f"{'Macro':8s} {fd.mean():10.4f} {fh.mean():12.4f} {delta.mean():+8.4f}")
    print("\nNote: positive Delta = accuracy lost on the hardware path.")
    print("      Delta ~ 0  = digital results are reproducible on device.")


if __name__ == "__main__":
    main()
