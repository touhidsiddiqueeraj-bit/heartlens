# HeartLens — Six-Experiment Framework

This file defines the experimental protocol for the WIECON paper.
Experiments 1–4 run on CPU/Colab; Experiment 5–6 run on the
ESP32-S3 (N16R8) via USB.

| # | Experiment | Where | Script / Mode | Status |
|---|------------|-------|---------------|--------|
| 1 | Patient-independent classification (grouped CV) | Colab | `group_kfold_eval.py` | ready |
| 2 | Noise robustness (Raw vs Filter vs AE, SNR 0–40 dB) | Colab | `evaluate_noise_robustness.py` | ready |
| 3 | External generalization (mitdb → SVDB + afdb) | Colab | `external_validation.py` | ready |
| 4 | Quantization (FP32 vs INT8 delta) | Colab | `train_classifier.py`, `compare_models.py` | ready |
| 5 | Embedded performance (latency, memory, size) | S3 board | `BENCHMARK_MODE` | needs board |
| 6 | Hardware-domain robustness (ΔF1) | S3 board | `REPLAY_MODE` + `hw_eval/` | needs board |

---

## Experiment 1 — Patient-independent classification

`GroupKFold` at the record (patient) level, repeated over 3 seeds,
5 folds. Report per-class F1 mean ± std and macro F1.

```bash
python3 group_kfold_eval.py --folds 5 --seeds 0,1,2 --epochs 40
# → results/group_kfold.json
```

## Experiment 2 — Noise robustness

Same patient-level test set, corrupted at SNR ∈ {0,5,10,15,20,30,40} dB,
classified through three front-ends:
1. **Raw** — noisy signal directly into the CNN
2. **Filter** — 0.5–45 Hz Butterworth bandpass, then CNN
3. **Autoencoder** — learned Conv1D denoiser, then CNN

```bash
python3 evaluate_noise_robustness.py --epochs 40
# → results/noise_robustness.json + .png
```

**Claim check (review #7):** if AE+CNN ≈ Raw+CNN across all SNR,
the learned denoiser adds no classification benefit — report honestly.

## Experiment 3 — External generalization

Train on MIT-BIH (patient-level split). Evaluate **without retraining** on:
- **SVDB** (supraventricular arrhythmia DB, 128 Hz → resampled 360 Hz):
  beat-level N/A/V labels — the true external test.
- **afdb** (atrial fibrillation DB): rhythm windows scored by the
  beat-level model — distribution check only, AF windows are not
  beat labels for this model.

```bash
python3 external_validation.py --epochs 40
# → results/external_validation.json
```

## Experiment 4 — Quantization

FP32 vs INT8 per-class F1 delta, computed inside `train_classifier.py`
and per-architecture in `compare_models.py` (int8 size, delta).

## Experiment 5 — Embedded performance

Board: ESP32-S3 N16R8. Flash with `BENCHMARK_MODE=1`:

```ini
[env:esp32-s3-devkitc-1]
build_flags = ${env.build_flags} -DBENCHMARK_MODE=1
```

Upload, open monitor, send `r` → the board prints total, denoise-avg,
classify-avg, per-window-avg microseconds over all sliding windows,
plus the resulting class/confidence. Repeat for each architecture by
swapping `models/*.h` (convert_tflite_to_headers.sh). Log results into
`hw_eval/captures/latency.csv` (model, size_kb, per_window_ms, total_ms).

Memory: report `Free heap` from the boot banner and model sizes.

## Experiment 6 — Hardware-domain robustness (ΔF1)

Signal path under test: **DAC (GPIO17) → jumper wire → ADC (GPIO4) → ESP32**.
This validates quantization + ADC + DMA sampling + firmware inference on
real silicon without an analog frontend. The AD8232 + electrode path is
documented future work (no frontend hardware available).

Steps:
1. Train + quantize classifier (Exp 1/4 outputs `classifier_int8.tflite`).
2. Flash with `REPLAY_MODE=1`; jumper GPIO17 → GPIO4.
3. Digital leg + drive the board:
   ```bash
   python3 hw_eval/replay_drive.py --port /dev/ttyACM0 \
       --model heart-lens-training/models/classifier_int8.tflite
   ```
4. Compute ΔF1:
   ```bash
   python3 hw_eval/compute_delta.py \
       --digital hw_eval/captures/digital.json \
       --hardware hw_eval/captures/hardware.json
   ```
   ΔF1_hardware = F1_digital − F1_hardware.

Protocol notes:
- Each 1-s segment is replayed 10× back-to-back (3600 samples ≈ 10 s
  nominal) so the full sliding-window pipeline runs per segment;
  wall time ≈ 1 s/segment (REPLAY_RATE_MULT=10).
- 8-bit DAC quantization is part of the measured domain shift — do
  not filter it out.
- Skipped/failed frames are reported; rerun with fewer segments if
  the skip rate is high.

---

## What is NOT in this study

- Real acquisition (electrodes → AD8232 → ESP32): requires frontend
  hardware + IRB — future work section of the paper.
- Battery life: requires current measurement (DMM/INA219) — protocol:
  charge fully, run continuous inference, log battery % every 30 s,
  report time to BAT_STOP_MV. Estimate table in README remains an estimate.
