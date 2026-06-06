# HeartLens AI

**Edge ECG Monitoring with On-Device AI Inference**

A fully offline, sub-$15 ECG screening device that runs AI inference directly on the hardware — no smartphone, no internet, no medical training required.

> *"Every year, millions of people suffer preventable heart attacks not because treatment does not exist, but because warning signs went unnoticed until too late. HeartLens AI is built to close that gap — for anyone, anywhere, at a cost lower than a restaurant meal."*

---

## Regulatory Scope

**HeartLens AI is a preventative screening aid, not a medical device.** It is not intended to diagnose, treat, or manage any condition. Its sole function is to flag rhythm patterns that may warrant professional evaluation. The device has not received FDA, CE, or any regulatory clearance. Use at your own risk.

**Single-lead ECG limitation:** This device uses a single ECG lead (Lead I equivalent). Clinical rhythm classification typically requires 12-lead ECG. Single-lead systems have limited diagnostic capability for complex arrhythmias, ST-segment analysis, and axis determination.

---

## Overview

| Parameter | Value |
|-----------|-------|
| Target BOM | ~$12 USD per unit |
| Connectivity | None — fully offline |
| MCU | ESP32-WROOM-32 (dual-core 240 MHz, 520 KB SRAM) |
| AI Models | Conv1D denoiser + 1D-CNN classifier (TFLite Micro, int8) |
| Detects | Normal, AFib (proxy), PVC |
| ADC Sampling | I2S-driven with esp_adc_cal calibration |
| Inference | Sliding window (50% overlap, majority voting) |
| Battery Life | ~6.5 hours (500 mAh LiPo) |
| Build Timeline | 16 weeks (part-time) |
| License | MIT |

---

## Repository Structure

```
heart-lens/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── build_proposal.js                  # Node.js script to generate proposal .docx
├── circuit_diagram.md                 # Mermaid circuit diagram source
├── circuit_diagram.svg                # Rendered system block diagram
├── todo.md                            # Full project to-do list (16 weeks)
│
├── HeartLens_Firmware/                # ESP32 Arduino firmware
│   ├── README.md                      # Firmware-specific documentation
│   ├── HeartLens_Firmware.ino         # Main entry point (v1.1)
│   ├── setup_tflite_micro.sh          # Downloads TFLite Micro source
│   ├── convert_tflite_to_headers.sh   # .tflite → C header (with extern C guards)
│   ├── src/
│   │   ├── Config.h                   # Pin mapping, thresholds, ADC calibration, WDT
│   │   ├── adc_sampler.h/.cpp         # I2S ADC sampling with mutex sync
│   │   ├── ecg_processor.h/.cpp       # TFLite Micro, sliding window inference
│   │   ├── interpreter.h/.cpp         # Confidence → plain-language output
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
│   ├── data_loader.py                 # WFDB parser, patient-level split
│   ├── noise_pipeline.py              # 4 noise types + SNR mixing
│   ├── train_denoiser.py              # Conv1D autoencoder training → int8 TFLite
│   ├── train_classifier.py            # 1D-CNN classification training → int8 TFLite
│   ├── quantize.py                    # Standalone int8 quantization
│   ├── generate_dummy_models.py       # Dummy models for firmware bringup
│   ├── models/
│   │   ├── denoiser_dummy.tflite      # 3.6 KB dummy denoiser
│   │   └── classifier_dummy.tflite    # 12.6 KB dummy classifier
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

### Key Improvements v1.1

| Area | Improvement | Impact |
|------|-------------|--------|
| **ADC** | I2S-driven sampling + `esp_adc_cal` calibration | Eliminates analogRead jitter, corrects voltage mapping |
| **Thread safety** | FreeRTOS mutex on shared buffer | Prevents data races between Core 0 (ADC) and Core 1 (inference) |
| **Task lifecycle** | Proper `vTaskDelete` on stopSampling | ADC task stops during inference |
| **Inference** | Sliding window (50% overlap) + majority voting | Uses full 10s window, not 1s |
| **Denoiser** | Conv1D autoencoder replaces LSTM | 3-5x faster on ESP32, 3x smaller model |
| **Dequantization** | Uses TFLite scale/zero_point correctly | Confidence values are proper probabilities |
| **Interpreter** | Working debounce, proper LOW < HIGH branches | No false alerts within 30s of normal |
| **Display** | I2C retry on init, dirty-flag battery render | Survives hung I2C, no flicker |
| **Battery** | 16-sample averaging + LiPo lookup table | ±2% accuracy vs ±15% before |
| **Lead-off** | Hysteresis on reconnection | No rapid state fluttering |
| **Watchdog** | `esp_task_wdt` configured explicitly | Hard crash recovery guarantee |

### Inference Pipeline

1. **ADC Sampling** (Core 0, I2S-driven 360 Hz) — 10-second window (3,600 int16 samples)
2. **Denoiser** (Conv1D autoencoder, int8, ~50 KB) — removes motion artifacts, baseline wander, PLI, EMG noise
3. **Classifier** (1D-CNN, int8, ~112 KB) — sliding window (50% overlap, majority vote)
4. **Interpreter** (rule-based) — confidence thresholds → 1 of 4 plain-language messages
5. **Display** (SSD1306 OLED) — shows result + battery indicator

### Plain-Language Outputs

| Condition | Message |
|-----------|---------|
| Normal | "Heart rhythm looks normal." |
| Medium urgency | "Unusual rhythm detected. Please see a doctor soon." |
| High urgency | "Irregular rhythm detected. Please seek medical attention." |
| Unclear signal | "Signal unclear. Please reattach electrodes and try again." |

### Quick Start

```bash
cd HeartLens_Firmware
./setup_tflite_micro.sh

# Open HeartLens_Firmware.ino in Arduino IDE
# Install: ESP32 board package, Adafruit SSD1306, Adafruit GFX
# Select Board: ESP32 Dev Module
# Upload
```

### After Training Real Models

```bash
./convert_tflite_to_headers.sh ../heart-lens-training/models models/
```

---

## Training Pipeline

### Dataset

The model is trained on the **MIT-BIH Arrhythmia Database** (PhysioNet) — 48 half-hour ECG recordings at 360 Hz.

**Label mapping (v1.1):** Only cleanly identifiable classes are used. Pathological proxy mappings have been removed (see [Clinical Label Audit](#clinical-label-audit)).

| Class | Name | Source Annotation |
|-------|------|-------------------|
| 0 | Normal Sinus Rhythm | `N` |
| 1 | Atrial Fibrillation (proxy) | `A` — atrial premature beat (see warning below) |
| 2 | Premature Ventricular Contraction | `V` |

**WARNING:** Class 1 (AFib) currently uses atrial premature beats (`A`) as a proxy. This is NOT clinically equivalent. The model will learn to detect APBs, not AFib. For production use, replace with properly annotated AFib data from [MIT-BIH AF Database](https://physionet.org/content/afdb/).

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
- Input: 360 samples × 1 channel (denoised)
- 3 Conv1D blocks: 32/64/128 filters, kernel 5, BatchNorm, MaxPool
- Global Average Pooling → Dense(64, ReLU) → Dropout(0.5) → Dense(6, Softmax)
- Quantized size: ~112 KB

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
2. **AFib is a proxy** — Currently detects APB patterns, not true AFib. Use MIT-BIH AF Database for production
3. **ESP32 internal ADC** — Still limited by ±6% non-linearity. Recommend external ADC (ADS1292R) for clinical-grade signal
4. **No real-time streaming** — Operates in 10-second analysis windows. Paroxysmal events between windows may be missed
5. **Regulatory** — No FDA/CE clearance. Not for diagnostic use

---

## Publication Targets

| Venue | Type | Impact Factor |
|-------|------|---------------|
| Sensors (MDPI) | Primary | Q2, IF ~3.9 |
| IEEE Internet of Things Journal | Alternate | Q1, IF ~10.6 |
| Biomedical Signal Processing & Control | Alternate | Q1, IF ~5.1 |
| IEEE EMBC | Conference backup | Top-tier |

---

## License

MIT — see [LICENSE](LICENSE).
