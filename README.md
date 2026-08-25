# HeartLens AI

**Comparative Evaluation of Deep Learning Models on Edge Devices for Detection of Abnormal Heart Rhythms**

A research prototype: a fully offline, sub-$15 ECG screening device that runs AI inference directly on the hardware — no smartphone, no internet. This repository is the experimental framework for a WIECON-ECE conference paper: six experiments (grouped cross-validation, noise robustness, external generalization, quantization, embedded latency, hardware-domain robustness) comparing CNN / LSTM / GRU / TCN architectures on an ESP32-S3.

> *Every year, millions of people experience arrhythmias whose early warning signs go unnoticed. HeartLens AI explores whether sub-$15 edge hardware can flag abnormal heart rhythms reliably enough to warrant professional evaluation — for anyone, anywhere, at a cost lower than a restaurant meal.*

---

## Regulatory Scope

**HeartLens AI is a research prototype.** It has not received FDA, CE, or any regulatory clearance, and the repository makes no regulatory classification claim. Its intended function is to flag rhythm patterns that may warrant professional evaluation; it is not intended to diagnose, treat, or manage any condition. Use at your own risk.

**Single-lead ECG limitation:** This device uses a single ECG lead (Lead I equivalent). Clinical rhythm classification typically requires 12-lead ECG. Single-lead systems have limited diagnostic capability for complex arrhythmias, ST-segment analysis, and axis determination.

---

## Overview

| Parameter | Value |
|-----------|-------|
| Target BOM | ~$12 USD per unit (estimate) |
| Connectivity | None — fully offline |
| MCU | ESP32-WROOM-32 (docs) / **ESP32-S3 N16R8 (verified target)** |
| AI Models | Conv1D denoiser + 1D-CNN classifier (TFLite Micro, int8) |
| Detects (beat-level) | Normal, APB, PVC |
| Detects (rhythm-level) | AF / Normal — via afdb loader (separate model) |
| ADC Sampling | ESP32: task-polled; ESP32-S3: DMA continuous mode |
| Inference | Sliding window (50% overlap, confidence-weighted voting) |
| Signal quality | Heuristic SQI gate before classification |
| Battery Life | ~6.5 hours (500 mAh LiPo) — **estimate, not measured** |
| Paper target | IEEE WIECON-ECE 2026 (conference) |

---

## Repository Structure

```
heart-lens/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies (auto-installed)
├── auto_train.py                      # End-to-end training pipeline (CLI + GUI)
├── circuit_diagram.md                 # Mermaid circuit diagram source
├── circuit_diagram.svg                # Rendered system block diagram
│
├── docs/
│   └── EXPERIMENTS.md                 # Six-experiment paper protocol
│
├── colab/
│   └── HeartLens_Colab.ipynb          # Zero-upload training + experiments
│
├── hw_eval/                           # Hardware-domain experiments (Exp 6)
│   ├── replay_drive.py                # Streams segments → board, digital leg
│   └── compute_delta.py               # ΔF1 = F1_digital − F1_hardware
│
├── HeartLens_Firmware/                # ESP32/ESP32-S3 Arduino firmware
│   ├── README.md                      # Firmware-specific documentation
│   ├── HeartLens_Firmware.ino         # Main entry point (v1.2)
│   ├── platformio.ini                 # Headless builds: esp32-s3-devkitc-1, esp32dev
│   ├── setup_tflite_micro.sh          # Downloads TFLite Micro source
│   ├── convert_tflite_to_headers.sh   # .tflite → C header (with extern C guards)
│   ├── src/
│   │   ├── Config.h                   # Pins, thresholds, modes, calibration, SQI
│   │   ├── adc_sampler.h/.cpp         # DMA continuous (S3) / legacy (ESP32) sampling
│   │   ├── ecg_processor.h/.cpp       # TFLite Micro, weighted voting, SQI gate
│   │   ├── interpreter.h/.cpp         # Calibrated confidence → plain-language output
│   │   ├── display.h/.cpp             # SSD1306 OLED with I2C error recovery
│   │   ├── battery.h/.cpp             # LiPo monitoring with lookup table
│   │   ├── lead_off.h/.cpp            # Electrode disconnect with hysteresis
│   │   └── debug.h                    # Toggleable serial debug
│   ├── models/
│   │   ├── denoiser_model.h           # Conv1D int8 quantized denoiser (C array)
│   │   └── classifier_model.h         # int8 quantized classifier (C array)
│   └── lib/tensorflow_lite/           # TFLite Micro source (25 MB, gitignored)
│
├── heart-lens-training/               # Python training pipeline
│   ├── requirements.txt               # Pinned dependencies
│   ├── TRAINING_DATA.md               # MIT-BIH dataset statistics
│   ├── data_loader.py                 # WFDB parser, 1 s windows, 3 classes
│   ├── afdb_loader.py                 # MIT-BIH AF DB rhythm windows (250→360 Hz)
│   ├── noise_pipeline.py              # 4 noise types + SNR mixing
│   ├── train_denoiser.py              # Conv1D autoencoder training → int8 TFLite
│   ├── train_classifier.py            # 3-class CNN training → int8 TFLite
│   ├── models.py                      # Shared builders: CNN / LSTM / GRU / TCN
│   ├── group_kfold_eval.py            # Exp 1: GroupKFold × seeds, mean±std
│   ├── evaluate_noise_robustness.py   # Exp 2: Raw/Filter/AE × SNR 0–40 dB
│   ├── external_validation.py         # Exp 3: mitdb → SVDB + afdb
│   ├── calibrate.py                   # Exp 4: temperature scaling → Config.h
│   ├── compare_models.py              # Exp 5: architecture comparison study
│   ├── quantize.py                    # Standalone int8 quantization
│   ├── generate_dummy_models.py       # Dummy models for firmware bringup
│   ├── models/                        # Trained .tflite/.keras checkpoints
│   ├── results/                       # JSON/CSV/PNG experiment outputs
│   └── mitdb/                         # MIT-BIH Arrhythmia Database (gitignored)
│
├── .github/workflows/                 # CI configuration
│   └── ci.yml                         # Firmware structure check + Python lint
│
└── HeartLens_AI_Project_Proposal.*    # Generated proposal document
```

---

## Hardware

### Pin Mapping

| ESP32 Pin | Connects To | Function |
|-----------|-------------|----------|
| GPIO34 | AD8232 OUTPUT | ECG signal (ADC1_CH6, I2S-driven) |
| GPIO32 | AD8232 LOFF+ | Lead-off detect positive |
| GPIO33 | AD8232 LOFF- | Lead-off detect negative |
| GPIO21 | SSD1306 SDA | I2C data (4.7k pull-up) |
| GPIO22 | SSD1306 SCL | I2C clock (4.7k pull-up) |
| GPIO35 | Battery divider (10k/10k) | LiPo voltage monitor |
| GPIO2 | Built-in LED | Status indicator |
| GPIO1/3 | UART | Debug serial (115200 baud) |

### Bill of Materials

| Component | Model | Cost |
|-----------|-------|------|
| Microcontroller | ESP32-WROOM-32 | $3.50 |
| ECG Analog Frontend | AD8232 breakout | $2.00 |
| Display | SSD1306 OLED 128×64 I2C | $1.50 |
| Electrodes | Snap cable + gel pads | $1.50 |
| Battery | LiPo 500–1000 mAh 3.7V | $1.50 |
| Charger | TP4056 USB-C | $0.50 |
| PCB | Custom 2-layer 60×40 mm | $0.40 |
| Passives | Resistors, caps, headers | $1.00 |
| **Total** | | **~$11.90** |

### Power Budget

| Component | Current |
|-----------|---------|
| ESP32 (active, 240 MHz) | ~60 mA |
| AD8232 | ~0.5 mA |
| SSD1306 (average) | ~10 mA |
| AMS1117 quiescent | ~5 mA |
| **Total** | **~75 mA** |
| **Battery life (500 mAh)** | **~6.5 hours** |

---

## Firmware

### State Machine

```
         ┌──────────────────────────────────────┐
         │                                      ▼
    ┌────────┐  lead_off   ┌──────────┐  full  ┌───────────┐
    │  IDLE  │───────────▶│ SAMPLING │──────▶│ INFERENCE │
    └────┬───┘            └──────────┘        └─────┬─────┘
         ▲                                          │
         │                ┌──────────┐               │
         │                │ DISPLAY  │◄──────────────┘
         │                └────┬─────┘
         │                     │ 10s timeout
         └─────────────────────┘
```

### Key Improvements v1.2

| Area | Improvement | Impact |
|------|-------------|--------|
| **Board support** | ESP32-S3 N16R8 target (PlatformIO), USB-CDC serial | Verified hardware, no UART needed |
| **ADC (S3)** | DMA continuous mode (`esp_adc_continuous`) | Hardware-timed sampling, replaces legacy polling |
| **Classifier** | 3 classes (Normal / APB / PVC), output dim matches training | Removed dead outputs (was Dense(6)) |
| **Voting** | Confidence-weighted (sum of softmax probs) instead of argmax majority | Low-confidence windows no longer count equally |
| **Signal quality** | Heuristic SQI gate (flat/saturation + HF-noise ratio) | Corrupted buffers never reach the classifier |
| **Calibration** | Temperature scaling (`calibrate.py` → `CALIB_TEMPERATURE`) | Confidence is a calibrated probability |
| **Denoiser** | Conv1D autoencoder replaces LSTM | 3-5x faster on ESP32, 3x smaller model |
| **Dequantization** | Uses TFLite scale/zero_point correctly | Confidence values are proper probabilities |
| **REPLAY_MODE** | Serial-streamed ECG → DAC playback → ADC → inference | Hardware-domain experiment (ΔF1) without analog frontend |
| **BENCHMARK_MODE** | Per-stage `micros()` timing on real silicon | Exp 5 latency numbers |

### Inference Pipeline

1. **ADC Sampling** (Core 0; ESP32: task-polled 360 Hz, ESP32-S3: DMA continuous 36 kHz → decimated) — 10-second window (3,600 samples)
2. **Signal quality gate** (heuristic) — flat/saturated/high-noise buffers rejected before classification
3. **Denoiser** (Conv1D autoencoder, int8, ~50 KB) — removes motion artifacts, baseline wander, PLI, EMG noise
4. **Classifier** (1D-CNN, int8, ~112 KB) — sliding window (50% overlap, confidence-weighted voting)
5. **Calibration** (temperature scaling) — confidence → calibrated probability
6. **Interpreter** (rule-based) — calibrated confidence thresholds → 1 of 4 plain-language messages
7. **Display** (SSD1306 OLED) — shows result + battery indicator

### Plain-Language Outputs

| Condition | Message |
|-----------|---------|
| Normal | "Heart rhythm looks normal." |
| Medium urgency | "Unusual rhythm detected. Please see a doctor soon." |
| High urgency | "Irregular rhythm detected. Please seek medical attention." |
| Unclear signal | "Signal unclear. Please reattach electrodes and try again." |

### Quick Start (PlatformIO, ESP32-S3)

```bash
cd HeartLens_Firmware
pio run -t upload -e esp32-s3-devkitc-1
pio device monitor -e esp32-s3-devkitc-1
```

Arduino IDE users: follow `setup_tflite_micro.sh`, select ESP32S3 Dev Module with USB CDC enabled.

### After Training Real Models

```bash
cd HeartLens_Firmware
./convert_tflite_to_headers.sh ../heart-lens-training/models models/
```

---

## Training Pipeline

### Automated End-to-End Pipeline

A single-script entry point (`auto_train.py`) orchestrates everything — dependency installation, dataset loading, denoiser + classifier training, model conversion, and PDF report generation.

```bash
# Auto-installs missing deps, trains both models, outputs PDF report
python3 auto_train.py --epochs 50 --max-per-class 3000

# GUI mode (uses Tkinter)
python3 auto_train.py --gui

# Skip denoiser / classifier for faster iteration
python3 auto_train.py --skip-denoiser --epochs 1 --max-per-class 20
```

**Auto-dep-install:** The script checks for `numpy`, `tensorflow`, `scikit-learn`, `matplotlib`, `fpdf2`, `scipy`, and `wfdb` on startup. Any missing package is installed via `pip` automatically before the training starts. No manual `pip install -r requirements.txt` needed.

**GUI mode (`--gui`):** Launches a Tkinter window with parameter controls (epochs, max-per-class, data directory, skip toggles), a real-time log output panel, an indeterminate progress bar, and Run/Cancel buttons. The training runs in a background thread so the UI stays responsive.

### Label Mapping (beat-level, 3 classes)

The model is trained on the **MIT-BIH Arrhythmia Database** (PhysioNet) — 48 half-hour ECG recordings at 360 Hz. Class 1 is **honestly labeled APB** (atrial premature beat); it is *not* atrial fibrillation. True rhythm-level AF data comes from the **MIT-BIH Atrial Fibrillation Database** via `afdb_loader.py` (see [Methodology](#methodology-beat-level-vs-rhythm-level)).

| Class | Name | Source Annotation |
|-------|------|-------------------|
| 0 | Normal Sinus Rhythm | `N` |
| 1 | Atrial Premature Beat (APB) | `A` |
| 2 | Premature Ventricular Contraction | `V` |

### Methodology: Beat-Level vs Rhythm-Level (review #2)

| Level | Window | Classes | Data |
|-------|--------|---------|------|
| Beat-level | 1 s (360 samples @ 360 Hz) | Normal, APB, PVC | mitdb beat annotations |
| Rhythm-level | 10 s | AF, Normal | afdb rhythm annotations (250 Hz → resampled 360 Hz) |

Beat-level events (PVC, APB) and rhythm-level conditions (AF) are deliberately **not mixed** into one classification task. The 10-second firmware inference buffer feeds 1-second beat-level windows through a sliding window; the rhythm-level AF detector is a separate model trained on afdb.

### Clinical Label Audit

The following proxy mappings present in the original codebase (v1.0) have been **removed**:

| Removed Mapping | Original Class | Reason |
|-----------------|----------------|--------|
| LBBB (`L`) → Normal | 0 | Pathological conduction abnormality |
| RBBB (`R`) → Normal | 0 | Pathological conduction abnormality |
| Ventricular flutter (`!`) → Tachy | 3 | Pre-arrest rhythm, distinct from SVT |
| Fusion beat (`F`) → Tachy | 3 | Non-specific |
| Paced (`/`) → Brady | 4 | Pacing is treatment, not rhythm |
| Fusion paced (`f`) → Brady | 4 | Same |
| Atrial escape (`E`) → ST Abn | 5 | Different mechanism |
| Nodal escape (`J`) → ST Abn | 5 | Different mechanism |
| Aberrated APB (`a`) → ST Abn | 5 | Different mechanism |
| Supraventricular premature (`S`) → ST Abn | 5 | Different mechanism |
| Nodal premature (`j`) → ST Abn | 5 | Different mechanism |

### Model Architectures

**Denoiser** (Conv1D autoencoder):
- Input: 360 samples × 1 channel (1 second at 360 Hz)
- Encoder: Conv1D(16,15) → MaxPool → Conv1D(8,15) → MaxPool
- Decoder: UpSample → Conv1D(8,15) → UpSample → Conv1D(1,15)
- Loss: MSE on clean ECG reconstruction
- Quantized size: ~50 KB
- **3-5x faster than LSTM on ESP32**

**Classifier** (1D-CNN):
- Input: 360 samples × 1 channel (1 second, denoised)
- 3 Conv1D blocks: 32/64/128 filters, kernel 5, BatchNorm, MaxPool
- Global Average Pooling → Dense(64, ReLU) → Dropout(0.5) → Dense(3, Softmax)
- Quantized size: ~112 KB
- Alternatives (comparison study): LSTM, GRU, TCN variants in `models.py`

### Training

```bash
cd heart-lens-training
python3 data_loader.py                  # Download MIT-BIH & extract segments
python3 train_denoiser.py               # Train Conv1D denoiser → int8 TFLite
python3 train_classifier.py             # Train classifier → int8 TFLite
./convert_tflite_to_headers.sh          # Update firmware C headers
```

### Patient-Level Cross-Validation

The data loader now splits at the **record level** (not segment level), preventing data leakage where the same patient appears in both train and test sets. This provides a more realistic estimate of real-world performance.

### Noise Augmentation

The noise pipeline generates 7 SNR levels (0, 5, 10, 15, 20, 30, 40 dB) for each clean segment:
- **Baseline wander** — 0.1–0.5 Hz low-frequency drift
- **Motion artifact** — random step/impulse events with exponential decay
- **Power-line interference** — 50 Hz sinusoid
- **EMG noise** — 20–100 Hz band-passed white noise

---

## Experiments (WIECON paper)

Six experiments, four runnable on Colab today, two on the ESP32-S3 via USB. Full protocols in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

| # | Experiment | Script / Mode | Status |
|---|------------|---------------|--------|
| 1 | Patient-independent classification (GroupKFold × 3 seeds) | `group_kfold_eval.py` | ready |
| 2 | Noise robustness (Raw vs Filter vs AE, SNR 0–40 dB) | `evaluate_noise_robustness.py` | ready |
| 3 | External generalization (mitdb → SVDB + afdb) | `external_validation.py` | ready |
| 4 | Quantization (FP32 vs INT8) | `train_classifier.py`, `compare_models.py` | ready |
| 5 | Embedded latency (4 architectures) | `BENCHMARK_MODE` | needs board |
| 6 | Hardware-domain ΔF1 (DAC→ADC replay) | `REPLAY_MODE` + `hw_eval/` | needs board |

## Run on Google Colab (no uploads)

```text
colab/HeartLens_Colab.ipynb   # clone → pip → train → experiments → zip
```

Datasets (mitdb/afdb/svdb) auto-download from PhysioNet inside Colab; code clones from this repo; results come back as a ~1 MB zip. Nothing needs to be uploaded at any point.

---

## Project Timeline (16 Weeks)

| Sprint | Weeks | Theme | Key Deliverable |
|--------|-------|-------|-----------------|
| Sprint 1 | W1–W3 | Hardware & Signal Acquisition | Prototype board, clean ECG on scope |
| Sprint 2 | W4–W7 | Data Pipeline & Model Training | Both models trained & quantized |
| Sprint 3 | W8–W10 | Firmware Integration | Full pipeline on ESP32 <100 ms |
| Sprint 4 | W11–W13 | Noise Study & Feasibility Testing | Real-world accuracy benchmarks |
| Sprint 5 | W14–W16 | Documentation & Publication | Paper submitted, code open-sourced |

---

## Known Limitations

1. **Single-lead ECG** — Cannot perform 12-lead diagnostics (ST-segment localization, axis deviation)
2. **APB ≠ AF** — The beat-level model detects premature atrial contractions, not atrial fibrillation. True AF is a separate rhythm-level model trained on afdb (10 s windows)
3. **ESP32 internal ADC** — ~±6% non-linearity (ESP32) / improved but uncharacterized on S3. Recommend external ADC (ADS1292R) for clinical-grade signal
4. **No real-time streaming** — Operates in 10-second analysis windows. Paroxysmal events between windows may be missed
5. **Regulatory** — Research prototype. No FDA/CE clearance. Not for diagnostic use
6. **Unmeasured claims** — Battery life, latency, and power figures are estimates; Exp 5/6 measure latency on silicon, battery remains unmeasured (protocol in EXPERIMENTS.md)
7. **8-bit DAC replay** — The hardware-replay experiment uses the S3's 8-bit DAC; the quantization it introduces is part of the measured domain shift

---

## Publication Targets

| Venue | Type | Notes |
|-------|------|-------|
| IEEE WIECON-ECE 2026 | Conference (primary) | 2-column IEEEtran, ~5 pages |
| Sensors (MDPI) | Journal (alternate) | Q2, IF ~3.9 |
| IEEE Internet of Things Journal | Journal (alternate) | Q1, IF ~10.6 |
| Biomedical Signal Processing & Control | Journal (alternate) | Q1, IF ~5.1 |

---

## License

MIT — see [LICENSE](LICENSE).
