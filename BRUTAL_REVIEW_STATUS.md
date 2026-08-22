# Brutal Review — Implementation Status (2026-08-22)

**Board:** ESP32-S3 N16R8, 240 MHz, serial /dev/ttyACM0, PlatformIO espressif32@6.9.0  
**Paper:** `paper/main.pdf` 6 pages, 651 KB, compiles clean (1 warning font)  
**Pipeline:** 11/11 steps `run_experiments.sh` DONE (all foreground, reboot-safe via `.markers/` + per-fold `*.tmp→*.json`)  

## Critical (C1–C6)

| # | Review ask | Status | Evidence |
|---|------------|--------|----------|
| **C1** Architecture comparison incomplete (class-level only, not arch-level) | **DONE** | `group_kfold_eval_v2.py` 5×2 identical folds (frozen `folds_5x2.json`, 48 records), 4 archs ×10 folds=40 ckpts. Table II `tab:gcv`: CNN 0.619±0.242 CI95 0.150, TCN 0.615±0.229, LSTM 0.582±0.096, GRU 0.526±0.170, with per-fold support 0–502. File: `heart-lens-training/results/group_kfold_{cnn,tcn,lstm,gru}.json` + `group_kfold_all.json` |
| **C2** APB unacceptable (0.148±0.296 buried) | **DONE** | 16-way ablation `scripts/apb_ablation.py` (cnn/tcn/lstm/gru × baseline/weighted/focal/balanced, single split 493/24/401, val-only selection). Best: `cnn weighted APB F1 0.7317 P0.88 R0.62 macro 0.838`, `cnn balanced 0.684/0.876`, `tcn baseline 0.700` already strong, LSTM/GRU all ≤0.26. Table III `tab:apb` full 16 rows, `results/apb_ablation.json` + `apb_ckpt/` |
| **C3** SVDB macro misleading (APB 0 counted) | **DONE** | Fix script `scripts/fix_svdb_metric.py`: `svdb_macro_nv = (0.6627+0.4621)/2 = 0.5624`, `svdb_macro_3class_biased=0.3749` kept but not interpreted. Table V `tab:svdb` now N/V-macro 0.562, APB `---` (0 windows). File: `results/external_validation.json` |
| **C4** Quant conclusions on one split | **DONE** | Paired FP32→INT8 on identical folds `scripts/paired_quant.py` 40 folds (10 per arch). Result: **INT8 degrades** `cnn Δ -0.210±0.164 disagree 33%`, `tcn Δ -0.238±0.165 disagree 40%`, LSTM/GRU `float32` only (TensorListStack, `SELECT_TF_OPS` fallback, not quantizable). Table IV `tab:compare` now paired, `results/paired_quant.json` + `_summary.json`. Single-split 0.588→0.677 is split-specific, documented. |
| **C5** Not real-time (3.74 s/window) | **DONE** | Measured on silicon via `BENCHMARK_MODE` (pyserial, no `pio monitor`). **Three models:** robust CNN 3792 ms (595+3197, total 72.06s), CNN 3792 ms, TCN 3611 ms (595+3016, total 68.62s) — ref kernels, RTF 3.6–3.8 (>1, not real-time). Reported as toolchain bottleneck, Xtensa path bounded (paper §5). Files: `hw_eval/captures/latency_all.json` + `latency.csv` + `HARDWARE_REPORT.md`, boot heap 148884 bytes, 200KB arena. |
| **C6** One synthetic smoke test only | **PARTIAL** | Functional smoke **DONE** (19 windows, valid=1, bench-wait heartbeat, 149KB heap). Systematic DAC→ADC replay **BLOCKED on S3**: `SOC_DAC_SUPPORTED=0`, `driver/dac.h: No such file`, S3 has no DAC (ESP32 classic does). Build fails with REPLAY_MODE=1. Documented as future work: needs MCP4725 external DAC or ESP32 classic. File: `HARDWARE_REPORT.md` + `paper/main.tex` Limitations. `hw_eval/replay_drive.py` protocol intact. |

## Major (M7–M15)

| # | Ask | Status | Evidence |
|---|-----|--------|----------|
| **M7** No full deployment trade-off | **DONE** | `scripts/generate_deployment_table.py` → `results/deployment_master.json/csv/md` + `tab:deploy`: model/prec/macro/CI/size/latency/throughput/RTF/arena/deployable, plus `latency_ms_per_window_measured` (3792/3611). |
| **M8** Why deploy CNN when TCN better? | **DONE** | Pareto `results/pareto.png` + `fig:pareto`: x=measured latency (3792 vs 3611), y=macro (0.619 vs 0.615 within CI), bubble=size (77.9 vs 70.1). TCN 5% faster/smaller, CNN 0.05 higher APB after mitigation — tie, both within CI95 overlap. Rule stated in Methods (Pareto). |
| **M9** Learned denoiser not worth it | **DONE** | `scripts/per_artifact_noise.py` 5 artifacts×7 SNR×3 front-ends + `denoiser_cost_benefit.md`: Butterworth wins 3/5 (BW/EMG/mixed), raw wins motion, AE only wins PLI low-SNR. Cost: Butterworth 0KB/5ms avg 0.670 vs AE 19KB/595ms avg 0.591 on mixed. Conclusion: remove denoiser from deployed pipeline (Fig per_artifact). |
| **M10** AWGN insufficient | **DONE** (or narrowed) | Same per-artifact run: `baseline_wander, motion, pli, emg, mixed` separated, not collapsed. File `results/per_artifact_noise.json/png`. If not fit deadline, claim already narrowed to AWGN+per-artifact, documented. |
| **M11** SQI gate unvalidated | **DONE** | `scripts/sqi_ablation.py` sweep thresholds 0.20–1.0: clean false-reject 62.6% at 0.35, corrupted reject 27.9%, downstream macro 0.727→0.643 on kept. Fig sqi + `results/sqi_ablation.json/png`. Verdict: gate as tuned not deployable, needs retuning. |
| **M12** Calibration underused | **DONE** | `scripts/generate_calibration_diagram.py` (val-only T=0.350, 30 epochs): ECE 0.405→0.101 (paper 0.389→0.088 on held-out), NLL 0.613→0.214, Brier 0.315→0.079, reliability diagram `fig_calib.png` + `calibration_extended.json` (bins, Brier, NLL). Abstention threshold on val left as coverage vs F1 (not tuned on test). |
| **M13** Weak statistics | **DONE** (minimal) | Paired Δ with CI95 (1.96*SD/√n): CNN 0.210±0.164 CI95 0.102, TCN 0.238±0.165 CI95 0.102, plus `tab:gcv` CI95 per model. Grouped CV n=10 per model, identical folds → paired. No p-value hacking, only 2 key comparisons (CNN vs TCN). |
| **M14** Related work gap not exposed | **DONE** | `tab:gap` added (Kiranyaz/Hannun/Davidson/ESP32 2023/This work × Pat-ind/Ext/Noise/Quant/MCU/Calib/Latency) + gap paragraph. Files: `results/prior_work_gap.md`. |
| **M15** Discussion repetitive/overclaims | **DONE** | Discussion cut ~40% (from ~450 words to ~180), `paper/main.pdf` **7→6 pages** (651KB). Removed `fig_confusion/gcv/noise/compare` to supplementary (ponytail comment), kept pareto/calibration/per-artifact/sqi as primary. Claims bounded: “functional feasibility, not real-time”, “expected 5-10× with Xtensa, not benchmarked”, “memory not limiter: 81/70KB+19KB flash, 200KB arena, 148.9KB heap”. |

## What remains (honest)

* **Hardware ΔF1 (C6 full):** Needs external MCP4725 on S3 GPIO17→GPIO4 or ESP32 classic board. Current report is smoke + latency only, per review’s “minimum acceptable: rename to functional smoke test” — we did.
* **Energy:** Not measured (no INA219/DMM). Marked future work, no invented values (per M7).
* **Second external dataset (C3 bonus):** Not added — SVDB fix is sufficient for 6-page; INCARTDB requires task redefinition.
* **Kernel optimization:** Measured reference kernels only; Xtensa build-flag path is deployment engineering, left as bounded future work (we report what default toolchain produces).

## Reproducibility (for checklist)

* Frozen folds `results/folds_5x2.json` (5×2, 48 records), 40 ckpts in `results/group_kfold_ckpt/`, 16 APB ckpts, 40 paired ckpts — all atomic `.tmp→.json`, reboot-safe (`PYTHONUNBUFFERED=1`, `tee` foreground).
* `run_experiments.sh` 11/11 DONE, `TOTAL=11` (was 10, added `brutal_synthesis`).
* Paper 6 pages: `paper/main.tex` 651KB, `paper/figures/fig_{calib,pareto,per_artifact,sqi}.png` copied, `paper/main.pdf` compiles clean (1 font warning).

## Commands to re-run

```bash
./run_experiments.sh  # all 11 steps skip (DONE), resumes folds if killed
cat heart-lens-training/results/progress.txt          # 100% (20/20)
cat heart-lens-training/results/apb_progress.txt      # 100% (16/16)
bash scripts/progress.sh                               # pipeline + ckpts
pio run -e esp32-s3-devkitc-1 -t upload               # S3 bench (already 3792/3611 measured)
```

*ponytail: global folds file, 5×2 not 5×3 to fit 12h; external ADC (MCP4725) if clinical domain shift needed; Xtensa kernels if RTF<1 required.*
