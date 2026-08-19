#!/usr/bin/env python3
"""Merge per-architecture compare_models outputs into one comparison.

Reads model_comparison_{cnn,lstm,gru,tcn}.json (produced by
compare_models.py --suffix _cnn etc.), merges rows in canonical order,
and writes model_comparison.json + model_comparison.csv + a summary plot.

Usage:
    python3 merge_comparison.py
"""

import csv
import json
import os
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results"
ORDER = ["cnn", "lstm", "gru", "tcn"]


def main():
    rows = []
    for m in ORDER:
        p = OUT_DIR / f"model_comparison_{m}.json"
        if not p.exists():
            print(f"missing: {p} (skip)")
            continue
        with open(p) as f:
            data = json.load(f)
        for r in data:
            if r["model"] == m:
                rows.append(r)
                break
        else:
            print(f"warning: no row for {m} in {p}")

    if not rows:
        print("no rows to merge — run compare_models.py per arch first")
        return 1

    with open(OUT_DIR / "model_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(OUT_DIR / "model_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

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
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'model_comparison.json'}, "
          f"{OUT_DIR / 'model_comparison.csv'}, "
          f"{OUT_DIR / 'model_comparison.png'}")
    for r in rows:
        i8 = f"{r['macro_f1_int8']:.4f}" if r["macro_f1_int8"] is not None else "N/A"
        d = f"{r['quant_delta']:+.4f}" if r["quant_delta"] is not None else "N/A"
        print(f"  {r['model']:4s}  float32={r['macro_f1_float32']:.4f}  "
              f"int8={i8}  delta={d}  size={r['size_kb']:.1f} KB  "
              f"({r['quant_type']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
