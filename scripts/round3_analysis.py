#!/usr/bin/env python3
"""Round-3 analysis: (a) baseline-vs-weighted paired deltas under grouped CV,
(b) QAT FP32/PTQ/QAT summary. Writes results/round3_analysis.json + markdown."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "heart-lens-training" / "results"

def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None

def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    return {"n": n, "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
            "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "t_df": float(t), "p": float(p), "cohens_dz": float(dz)}

out = {"complete": True, "baseline_vs_weighted": {}, "qat": {}}

# (a) baseline (unweighted) vs weighted (paper Table II arm)
for m in ["cnn", "tcn"]:
    base = load(OUT / f"group_kfold_{m}_baseline.json")
    wght = load(OUT / f"group_kfold_{m}.json")
    if not base or not wght:
        print(f"[analysis] skip {m} (missing arm)"); continue
    b = {r["seed"] * 10 + r["fold"]: r for r in base["per_fold"]}
    w = {r["seed"] * 10 + r["fold"]: r for r in wght["per_fold"]}
    keys = sorted(set(b) & set(w))
    apb_b = [b[k]["f1"][1] for k in keys]; apb_w = [w[k]["f1"][1] for k in keys]
    mac_b = [b[k]["macro"] for k in keys]; mac_w = [w[k]["macro"] for k in keys]
    out["baseline_vs_weighted"][m] = {
        "n": len(keys),
        "weighted_macro": float(np.mean(mac_w)), "baseline_macro": float(np.mean(mac_b)),
        "weighted_apb": float(np.mean(apb_w)), "baseline_apb": float(np.mean(apb_b)),
        "delta_apb": paired(apb_w, apb_b),
        "delta_macro": paired(mac_w, mac_b),
    }
    d = out["baseline_vs_weighted"][m]
    print(f"[{m}] weighted macro {d['weighted_macro']:.4f} vs baseline {d['baseline_macro']:.4f} | APB {d['weighted_apb']:.4f} vs {d['baseline_apb']:.4f} | dAPB {d['delta_apb']['mean_diff']:+.4f}±{d['delta_apb']['ci95']:.4f} p={d['delta_apb']['p']:.4f} | dMacro {d['delta_macro']['mean_diff']:+.4f}±{d['delta_macro']['ci95']:.4f} p={d['delta_macro']['p']:.4f}")

# (b) QAT
import pathlib
q = load(OUT / "qat_cnn_summary.json")
if q:
    out["qat"]["cnn"] = {k: q[k] for k in ["n", "fp32_macro", "ptq_macro", "qat_macro",
                                           "delta_ptq_vs_fp32", "delta_qat_vs_fp32", "delta_qat_vs_ptq",
                                           "ci95_qat_vs_ptq", "disagree_ptq_vs_fp32", "disagree_qat_vs_fp32",
                                           "disagree_qat_vs_ptq", "size_kb_ptq", "size_kb_qat"]}
    print(f"[qat-cnn] FP32 {q['fp32_macro']:.4f} PTQ {q['ptq_macro']:.4f} QAT {q['qat_macro']:.4f} | dQAT-PTQ {q['delta_qat_vs_ptq']:+.4f}±{q['ci95_qat_vs_ptq']:.4f}")

tmp = OUT / "round3_analysis.json.tmp"
with open(tmp, "w") as f:
    json.dump(out, f, indent=2)
    f.flush()
    import os; os.fsync(f.fileno())
import os
os.replace(tmp, OUT / "round3_analysis.json")
print(f"saved {OUT/'round3_analysis.json'}")

# markdown sidecar
md_path = OUT / "round3_analysis.md"
with open(md_path, "w") as f:
    f.write("# Round-3 analysis\n\n")
    for m, d in out["baseline_vs_weighted"].items():
        f.write(f"## {m.upper()} weighted vs baseline (n={d['n']})\n")
        f.write(f"- weighted macro {d['weighted_macro']:.4f} vs baseline {d['baseline_macro']:.4f}\n")
        f.write(f"- weighted APB {d['weighted_apb']:.4f} vs baseline {d['baseline_apb']:.4f}\n")
        f.write(f"- dAPB {d['delta_apb']['mean_diff']:+.4f} ± {d['delta_apb']['ci95']:.4f} (p={d['delta_apb']['p']:.3f}, dz={d['delta_apb']['cohens_dz']:.2f})\n")
        f.write(f"- dMacro {d['delta_macro']['mean_diff']:+.4f} ± {d['delta_macro']['ci95']:.4f} (p={d['delta_macro']['p']:.3f}, dz={d['delta_macro']['cohens_dz']:.2f})\n")
    if "cnn" in out["qat"]:
        q = out["qat"]["cnn"]
        f.write(f"\n## QAT CNN (n={q['n']})\n")
        f.write(f"- FP32 {q['fp32_macro']:.4f} PTQ {q['ptq_macro']:.4f} QAT {q['qat_macro']:.4f}\n")
print(f"saved {md_path}")
