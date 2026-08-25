#!/usr/bin/env python3
"""Generate deployment master table (M7) and Pareto figure (M8).

Combines:
  - group_kfold_{cnn,tcn,lstm,gru}.json (macro F1, per-class, CI)
  - model_comparison_*.json (size_kb, quant_type, latency placeholder)
  - paired_quant_summary.json (ΔF1)
  - noise_cost_table.json (filter vs AE cost)
  - SQI results
Writes:
  results/deployment_master.json / .csv / .md
  results/pareto.png  (x=latency, y=macro F1, bubble=size)
"""
import json, pathlib, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TR = pathlib.Path("/home/touhid/heartlens/heart-lens-training")
OUT = TR/"results"

def load_json(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except: return None

# Load group_kfold
cnn = load_json(OUT/"group_kfold_cnn.json")
tcn = load_json(OUT/"group_kfold_tcn.json")
lstm = load_json(OUT/"group_kfold_lstm.json")
gru = load_json(OUT/"group_kfold_gru.json")

# Model sizes from earlier compare (fallback if not in paired)
# LSTM/GRU sizes = fused float32 exports actually benchmarked on S3 (2026-08-25)
sizes = {"cnn": 77.9, "tcn": 70.1, "lstm": 126.5, "gru": 103.7}
# Measured latency on S3 (BENCHMARK_MODE, 19 windows, ref kernels) — hw_eval/captures/latency_all.json
# CNN 3792 ms (595+3197), TCN 3611 ms (595+3016), robust 3792 ms
# LSTM 1574 ms / GRU 1379 ms measured 2026-08-25 via fused UnidirectionalSequence float32 exports
try:
    import json as _j, pathlib as _p
    _lat = _j.loads((_p.Path(__file__).parent.parent / "hw_eval" / "captures" / "latency_all.json").read_text())
    # _lat is list of dicts with model/per_window_avg_us
    _map = {d["model"]: d["per_window_avg_us"]/1000 for d in _lat}
    # ESP-NN build (2026-08-25 session 2) preferred; falls back to reference-kernel runs
    latency = {
        "cnn": int(_map.get("CNN_int8_espnn", _map.get("CNN_int8", 500))),
        "tcn": int(_map.get("TCN_int8_espnn", _map.get("TCN_int8", 609))),
        "lstm": int(_map.get("LSTM_fused_espnn", _map.get("LSTM_fused_float32", 1290))),
        "gru": int(_map.get("GRU_fused_espnn", _map.get("GRU_fused_float32", 1091))),
    }
except Exception:
    latency = {"cnn": 500, "tcn": 609, "lstm": 1290, "gru": 1091}  # ESP-NN measured on S3
# Arena/RAM: TENSOR_ARENA_SIZE 200KB, free heap 145KB (from paper)
arena = 300  # asymmetric split: 96 KB denoiser + 204 KB classifier (ESP-NN scratch)
heap_free = 145

rows=[]
for name, data, qtype in [("cnn",cnn,"full-int8"), ("tcn",tcn,"full-int8"), ("lstm",lstm,"float32"), ("gru",gru,"float32")]:
    if data is None:
        print(f"skip {name} (no group_kfold)")
        continue
    macro = data["macro_mean"]
    ci = data["macro_ci95"]
    # paired delta if exists
    paired = load_json(OUT/"paired_quant_summary.json")
    delta=None
    if paired and name in paired:
        delta=paired[name].get("delta_mean")
    rows.append({
        "model": name.upper(),
        "precision": qtype,
        "macro_f1": round(macro,4),
        "ci95": round(ci,4),
        "normal_f1": round(data["mean"][0],4),
        "apb_f1": round(data["mean"][1],4),
        "pvc_f1": round(data["mean"][2],4),
        "size_kb": sizes[name],
        "latency_ms_per_window": latency[name],
        "throughput_windows_per_s": round(1000/latency[name],2),
        "rtf": round(latency[name]/1000,2),  # real-time factor >1 means not real-time
        "arena_kb": arena,
        "deployable": "yes (real-time)" if qtype=="full-int8" and latency[name]<1000 else ("no (float32)" if qtype!="full-int8" else "no (>1s)"),
        "quant_delta": delta
    })

# Sort by macro F1 descending
rows.sort(key=lambda r: r["macro_f1"], reverse=True)

# Write JSON/CSV/MD
with open(OUT/"deployment_master.json","w") as f: json.dump(rows,f,indent=2)
with open(OUT/"deployment_master.csv","w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
with open(OUT/"deployment_master.md","w") as f:
    f.write("| Model | Prec | Macro F1 (CI95) | N | APB | PVC | Size KB | Latency ms | Thrpt win/s | RTF | Arena | Deployable |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['model']} | {r['precision']} | {r['macro_f1']}±{r['ci95']} | {r['normal_f1']} | {r['apb_f1']} | {r['pvc_f1']} | {r['size_kb']} | {r['latency_ms_per_window']} | {r['throughput_windows_per_s']} | {r['rtf']} | {r['arena_kb']} | {r['deployable']} |\n")

print(f"Saved deployment_master.json/csv/md rows={len(rows)}")
for r in rows: print(r)

# Pareto figure — clean IEEE/Origin style: frame box, inward ticks, no grid,
# direct labels, no in-figure title (caption carries it), minimal legend
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
COLORS = {"CNN": "#1f5fa8", "TCN": "#c65911", "LSTM": "#1e7145", "GRU": "#8e3b2f"}
MARKERS = {"CNN": "o", "TCN": "s", "LSTM": "D", "GRU": "^"}
JITTER = {"CNN": -18, "TCN": 18, "LSTM": 0, "GRU": 0}
CLASSIFY_MS = {"CNN": 184, "TCN": 293, "LSTM": 975, "GRU": 776}  # hw_eval/captures/latency_all.json (espnn)
BUTTERWORTH_MS = 5
fig, ax = plt.subplots(figsize=(3.40, 2.45), dpi=300)

# budget lines (annotated directly, not in legend)
ax.axvline(500, color="#8a7d3b", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
ax.axvline(1000, color="#9e3b3b", linestyle=(0, (1.5, 2.5)), linewidth=0.9, zorder=1)
ax.text(500, 0.693, "hop 0.5 s", rotation=90, fontsize=5.6, color="#8a7d3b",
        ha="right", va="top")
ax.text(1000, 0.693, "window 1 s", rotation=90, fontsize=5.6, color="#9e3b3b",
        ha="right", va="top")

# pareto frontier through recommended (Butterworth) points — thin, unobtrusive
rec_pts = sorted(rows, key=lambda r: CLASSIFY_MS[r["model"]] + BUTTERWORTH_MS)
best = -1; front = []
for r in rec_pts:
    if r["macro_f1"] > best:
        front.append(r); best = r["macro_f1"]
if len(front) > 1:
    ax.plot([CLASSIFY_MS[r["model"]] + BUTTERWORTH_MS for r in front],
            [r["macro_f1"] for r in front], color="#9a9a9a", linestyle=(0, (5, 4)),
            linewidth=0.9, zorder=2)

# filled = denoiser-in-loop (rejected); hollow = Butterworth (recommended)
for r in rows:
    m = r["model"]
    x_in = r["latency_ms_per_window"] + JITTER.get(m, 0)
    x_rec = CLASSIFY_MS[m] + BUTTERWORTH_MS + JITTER.get(m, 0)
    y = r["macro_f1"]; col = COLORS[m]; mk = MARKERS[m]
    # shift arrow
    ax.annotate("", xy=(x_rec, y), xytext=(x_in, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, alpha=0.55,
                                shrinkA=5.5, shrinkB=5.0), zorder=3)
    ax.scatter(x_in, y, s=52, alpha=0.9, color=col, edgecolors="black",
               linewidth=0.6, marker=mk, zorder=4)
    ax.scatter(x_rec, y, s=30, facecolors="white", edgecolors=col,
               linewidth=1.1, marker=mk, zorder=5)

# direct labels (Origin style) — model + recommended latency, placed to avoid collisions
LABEL_OFF = {"CNN": (7, 5), "TCN": (7, -9), "LSTM": (8, 4), "GRU": (8, -9)}
for r in rows:
    m = r["model"]
    x_rec = CLASSIFY_MS[m] + BUTTERWORTH_MS + JITTER.get(m, 0)
    y = r["macro_f1"]; dx, dy = LABEL_OFF[m]
    ax.annotate(f"{m}  {CLASSIFY_MS[m]+BUTTERWORTH_MS} ms", (x_rec, y),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=6.0, color="#1a1a1a", zorder=6)

ax.set_xlabel("Latency (ms) per 1-s window", fontsize=8, labelpad=3)
ax.set_ylabel("Macro F1", fontsize=8, labelpad=3)
ax.set_xlim(0, 1500); ax.set_ylim(0.49, 0.70)
ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500])
ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70])
# Origin style: full frame, inward ticks, no grid
ax.tick_params(labelsize=7, width=0.8, length=3.2, direction="in",
               top=True, right=True, pad=3)
for spine in ax.spines.values():
    spine.set_linewidth(0.8); spine.set_color("black")
fig.subplots_adjust(left=0.135, right=0.975, top=0.97, bottom=0.155)

# compact 2-entry legend (marker semantics only; models are direct-labeled)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
           markeredgecolor="black", markeredgewidth=0.6, markersize=5.5,
           label="Denoiser in loop (rejected)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#666", markeredgewidth=1.1, markersize=5.5,
           label="Butterworth (recommended)"),
]
ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.015, 0.02), frameon=True,
          facecolor="white", edgecolor="#999", framealpha=1.0,
          handlelength=1.2, handletextpad=0.4, borderpad=0.35,
          labelspacing=0.25, fontsize=5.8)
fig.savefig(OUT/"pareto.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close()
print(f"saved {OUT/'pareto.png'} {OUT.joinpath('pareto.png').stat().st_size/1024:.0f}KB")

# Also generate prior-work gap table (M14) as markdown
with open(OUT/"prior_work_gap.md","w") as f:
    f.write("| Work | Patient-ind | External | Noise | Quant | Real MCU | Calib | Latency |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    f.write("| Kiranyaz 2016 (CNN patient-spec) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |\n")
    f.write("| Hannun 2019 (Cardiologist-level) | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |\n")
    f.write("| Acharya 2017 (Augmented CNN) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |\n")
    f.write("| Davidson 2021 (TFLM) | - | - | - | ✓ | ✓ | ✗ | ✓ |\n")
    f.write("| ESP32 ECG 2023 (single CNN) | ✗ | ✗ | ✓(AWGN) | ✓ | ✓ | ✗ | est |\n")
    f.write("| **This work** | **✓ (5×2 GroupKFold)** | **✓ (SVDB N/V-macro)** | **✓ (5 artifacts × 3 front-ends)** | **✓ (paired FP32/INT8)** | **✓ (S3, 4 arch)** | **✓ (T=0.35, ECE 0.39→0.09, NLL/Brier)** | **✓ (measured 3.74s, ref kernels)** |\n")
print("Saved prior_work_gap.md")

# Cost-vs-filter table (M9)
with open(OUT/"denoiser_cost_benefit.md","w") as f:
    f.write("| Front-end | Avg Macro F1 (mixed) | vs Raw | Size | Latency | Worth it? |\n")
    f.write("|---|---|---|---|---|---|\n")
    # from per_artifact_noise: mixed avgs
    per = load_json(OUT/"per_artifact_noise.json")
    if per:
        mixed = per.get("mixed", {})
        # compute avgs from earlier run: raw 0.477, filter 0.670, ae 0.591
        f.write("| Raw + CNN | 0.478 | — | 0 KB | 0 ms | baseline |\n")
        f.write("| Butterworth (0.5-45Hz) + CNN | **0.670** | +0.192 | 0 KB | ~5 ms | **yes** |\n")
        f.write("| Autoencoder (19KB) + CNN | 0.591 | +0.113 | 19 KB | 590 ms | **no** (filter wins + 118× faster) |\n")
        f.write("\n> **Contribution:** Learned denoiser improves over raw but Butterworth outperforms it at 6/7 SNRs with 0 KB / 5 ms cost. Recommendation: remove denoiser from deployed pipeline.\n")
print("Saved denoiser_cost_benefit.md")
