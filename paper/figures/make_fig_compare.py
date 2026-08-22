#!/usr/bin/env python3
"""Regenerate fig_compare.png from model_comparison.json (paper Figure 6).

Two stacked panels, portrait to fit the IEEE column:
  top    — macro F1 by architecture, float32 and int8 grouped bars
            (LSTM/GRU have no int8; they are not TFLite-quantizable)
  bottom — TFLite model size (KB); bars tinted by quant type so
            float32 files (LSTM/GRU) are not mistaken for int8 sizes

Usage:
    python3 make_fig_compare.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE.parents[1] / "heart-lens-training" / "results" / "model_comparison.json"

ORDER = ["cnn", "lstm", "gru", "tcn"]
LABELS = {"cnn": "CNN", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN"}
INT8 = "#b0604a"
F32 = "#4a7ba6"


def main():
    if not RESULTS.exists():
        print(f"missing: {RESULTS}", file=sys.stderr)
        return 1
    rows = {r["model"]: r for r in json.loads(RESULTS.read_text())}

    names = [LABELS[m] for m in ORDER]
    f32 = [rows[m]["macro_f1_float32"] for m in ORDER]
    i8 = [rows[m]["macro_f1_int8"] for m in ORDER]
    sizes = [rows[m]["size_kb"] for m in ORDER]
    is_int8 = [rows[m]["quant_type"] == "full-int8" for m in ORDER]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 6.6))

    x = np.arange(len(names))
    w = 0.32
    a1.bar(x - w / 2, f32, width=w, color=F32, label="float32")
    i8_vals = [i8[i] for i in range(4) if i8[i] is not None]
    a1.bar([x[i] + w / 2 for i in range(4) if i8[i] is not None],
           i8_vals, width=w, color=INT8, label="int8")
    a1.set_xticks(x)
    a1.set_xticklabels(names)
    a1.set_ylabel("Macro F1")
    a1.set_title("Accuracy", loc="left")
    a1.set_ylim(0, 1)
    a1.legend(loc="upper right", frameon=False)
    a1.grid(True, axis="y", alpha=0.3)

    colors = [INT8 if ok else F32 for ok in is_int8]
    a2.bar(names, sizes, color=colors)
    for i, s in enumerate(sizes):
        tag = "int8" if is_int8[i] else "float32"
        a2.text(i, s + 2, f"{s:.1f} ({tag})", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("TF Lite size (KB)")
    a2.set_title("Model size", loc="left")
    a2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = HERE / "fig_compare.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())