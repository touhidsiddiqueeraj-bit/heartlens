# HeartLens AI — Project To-Do List

**Duration**: 16 weeks (part-time, ~10-15 hrs/week)
**Target BOM**: ~$12/unit
**Total Budget**: ~$62 (3 prototypes)

---

## Sprint 0 — Setup & Procurement (Before Week 1)

- [ ] Install toolchain: Arduino IDE + ESP32 board package
- [ ] Install Python ML stack: TensorFlow, Keras, NumPy, SciPy, scikit-learn, WFDB
- [ ] Install KiCad for PCB design
- [ ] Order components from AliExpress / LCSC / Digi-Key
  - [ ] ESP32-WROOM-32 dev boards (x3)
  - [ ] AD8232 breakout boards (x3)
  - [ ] SSD1306 OLED 0.96" I2C (x3)
  - [ ] TP4056 USB-C charge modules (x3)
  - [ ] AMS1117-3.3 regulators (x10)
  - [ ] LiPo 800 mAh batteries (x3)
  - [ ] Electrode snap cables + gel pads
  - [ ] Resistor/capacitor assortments
  - [ ] JST connectors, headers, perfboard
- [ ] Download MIT-BIH Arrhythmia Database from PhysioNet
- [ ] Verify ESP32 can flash Blink sketch

---

## Sprint 1 — Hardware & Signal Acquisition (Weeks 1-3)

### Week 1 — KiCad Schematic + PCB Layout
- [ ] Design AD8232 frontend circuit in KiCad
- [ ] Add ESP32 minimum circuit (decoupling caps, EN pull-up, UART header)
- [ ] Add SSD1306 I2C interface with pull-ups
- [ ] Add TP4056 + AMS1117 power tree
- [ ] Add battery voltage divider on GPIO35
- [ ] Add lead-off detection connections to GPIO32/33
- [ ] Route 2-layer PCB (60x40 mm)
- [ ] Generate Gerber files

### Week 2 — PCB Order + Component Sourcing
- [ ] Order PCBs from JLCPCB (5 boards, ~$10)
- [ ] Receive and inspect components
- [ ] Prepare breadboard prototype as backup
- [ ] Solder AD8232 breakout header pins
- [ ] Assemble power circuit on breadboard: TP4056 → LiPo → AMS1117 → 3.3V rail
- [ ] Verify 3.3V output with multimeter

### Week 3 — PCB Assembly + Signal Verification
- [ ] Solder boards (if PCBs arrived) or assemble on perfboard
- [ ] Connect electrodes to AD8232 inputs
- [ ] Connect AD8232 OUTPUT to oscilloscope probe
- [ ] Verify clean ECG waveform visible on scope
- [ ] Measure signal SNR (>20 dB target)
- [ ] Test lead-off detection circuit
- [ ] **Exit Gate**: Clean ECG on oscilloscope, SNR >20 dB, baseline stable

---

## Sprint 2 — Data Pipeline & Model Training (Weeks 4-7)

### Week 4 — Data Ingestion + Noise Pipeline
- [ ] Write Python script to parse MIT-BIH .dat/.hea files via WFDB
- [ ] Extract beat-labeled segments at 360 Hz native rate
- [ ] Build noise injection pipeline:
  - [ ] Motion artifact generator
  - [ ] Baseline wander (0.1-0.5 Hz sinusoid)
  - [ ] 50/60 Hz power-line interference
  - [ ] Muscle noise (EMG) simulation
  - [ ] Mix noise at SNR levels 0-40 dB
- [ ] Create 5x augmented dataset for denoiser training
- [ ] Split into train/val/test (70/15/15)
- [ ] Normalize all signals to [-1, 1]

### Week 5 — LSTM Denoiser Training
- [ ] Build encoder-decoder LSTM in Keras (2×64 units)
- [ ] Train on clean → noisy reconstruction (MSE loss)
- [ ] Monitor validation loss, apply early stopping
- [ ] Evaluate denoising on held-out test set
- [ ] Visualize sample reconstructions (before/after)
- [ ] Export model to SavedModel format

### Week 6 — CNN Classifier Training
- [ ] Build 1D-CNN in Keras (3 conv blocks: 32/64/128)
- [ ] Train on denoised clean segments (6-class softmax)
- [ ] Apply class-weighting for AFib/PVC minority classes
- [ ] Generate confusion matrix on test set
- [ ] Compute per-class precision, recall, F1
- [ ] Target: overall F1 > 0.87

### Week 7 — Quantization + Export
- [ ] Build representative calibration dataset (500 samples/class)
- [ ] Apply post-training int8 quantization to both models
- [ ] Verify size: LSTM < 160 KB, CNN < 120 KB
- [ ] Benchmark accuracy delta (quantized vs. float)
- [ ] If degradation >5%, apply QAT and retry
- [ ] Export both models as .tflite files
- [ ] **Exit Gate**: Classifier F1 > 0.87 on clean MIT-BIH test set

---

## Sprint 3 — Firmware Integration (Weeks 8-10)

### Week 8 — ADC Sampling + TFLite Micro
- [ ] Set up FreeRTOS tasks on ESP32:
  - [ ] Core 0: ADC sampling loop at 360 Hz
  - [ ] Core 1: Inference + display
- [ ] Implement circular buffer for 3600-sample window
- [ ] Integrate TFLite Micro runtime for ESP32
- [ ] Load quantized LSTM denoiser model
- [ ] Load quantized 1D-CNN classifier model
- [ ] Verify both models load into SRAM (check total usage)
- [ ] Run first inference on dummy data, verify output tensor shape

### Week 9 — Interpreter + Display + Integration
- [ ] Build C++ rule interpreter:
  - [ ] Map class to urgency tier and message string
  - [ ] Implement confidence thresholds (>0.75, 0.55-0.75, <0.55)
  - [ ] Handle edge case: all classes below threshold
- [ ] Drive SSD1306 OLED via I2C (Adafruit SSD1306 library)
- [ ] Render 4 possible plain-language outputs
- [ ] Show battery level indicator on display
- [ ] Wire full pipeline: ADC → denoise → classify → interpret → display

### Week 10 — Latency Benchmarking + Optimization
- [ ] Measure per-stage latency with `micros()` timestamps:
  - [ ] ADC window collection
  - [ ] LSTM denoising
  - [ ] CNN classification
  - [ ] Interpreter + display render
- [ ] Verify total pipeline latency < 100 ms
- [ ] Measure total SRAM usage (heap + stack + models)
- [ ] Profile ADC noise floor with ESP32 internal ADC
- [ ] If latency exceeds target:
  - [ ] Reduce window size to 1800 samples
  - [ ] Enable ESP32 240 MHz mode
  - [ ] Move display to separate task with lower priority
- [ ] **Exit Gate**: Full pipeline latency < 100 ms on hardware

---

## Sprint 4 — Noise Study & Feasibility Testing (Weeks 11-13)

### Week 11 — Real-World Noise Capture
- [ ] Design noise capture protocol (IRB checklist if needed)
- [ ] Capture real-world noise conditions:
  - [ ] Sitting still (baseline)
  - [ ] Walking
  - [ ] Arm movement
  - [ ] Typing / desk work
- [ ] Record raw ADC data to serial + save to PC
- [ ] Compare real noise statistics to synthetic noise
- [ ] Run inference on real-world captures
- [ ] Document synthetic vs. real-noise accuracy gap

### Week 12 — Participant Feasibility Study (n=10)
- [ ] Obtain ethics waiver / IRB approval (if needed)
- [ ] Recruit 10 healthy volunteers
- [ ] For each participant:
  - [ ] Attach 3 electrodes (chest or wrist placement)
  - [ ] Capture 5 one-minute recordings per placement
  - [ ] Log device output and confidence
  - [ ] Record reference using second device if available
- [ ] Collect all data to SD card or serial log

### Week 13 — Analysis + Benchmarks
- [ ] Build confusion matrix from participant data
- [ ] Compute per-class F1, precision, recall
- [ ] Compare accuracy: real-world vs. clean MIT-BIH
- [ ] Verify accuracy within 8% of clean benchmark
- [ ] Run McNemar test for statistical comparison
- [ ] Document all results in draft format
- [ ] **Exit Gate**: On-device accuracy within 8% of clean benchmark

---

## Sprint 5 — Documentation & Publication (Weeks 14-16)

### Week 14 — Paper Drafting
- [ ] Write Introduction / Background section
- [ ] Write Methods section (hardware + software)
- [ ] Write Results section (tables + figures)
- [ ] Generate figures:
  - [ ] System block diagram
  - [ ] Signal pipeline diagram
  - [ ] Confusion matrices (clean + real-world)
  - [ ] Accuracy vs. SNR curves
  - [ ] Latency breakdown bar chart
- [ ] Write Discussion / Conclusion section
- [ ] Compile bibliography (Zotero / BibTeX)
- [ ] Full paper draft complete

### Week 15 — Open-Source Release
- [ ] Create GitHub repository
- [ ] Upload KiCad files (schematic + PCB layout)
- [ ] Upload firmware (Arduino C++ project)
- [ ] Upload model training scripts (Python)
- [ ] Upload quantized .tflite model files
- [ ] Write README with build instructions
- [ ] Add LICENSE (MIT / GPL / CERN-OHL)
- [ ] Add bill of materials with sourcing links
- [ ] Tag release v1.0.0

### Week 16 — Submission
- [ ] Final paper review by peers (if possible)
- [ ] Format for target venue (Sensors / IEEE IoT Journal)
- [ ] Write cover letter to editor
- [ ] Submit via online portal
- [ ] Prepare supplementary materials (if required)
- [ ] **Exit Gate**: Paper submitted to target venue

---

## Milestone Gates Summary

| Gate | Sprint | Criteria | Fallback |
|------|--------|----------|----------|
| G1 | Sprint 1 | Clean ECG on scope, SNR >20 dB | Add INA333 instrumentation amp stage |
| G2 | Sprint 2 | Classifier F1 > 0.87 on clean test set | Apply QAT, increase training epochs |
| G3 | Sprint 3 | Full pipeline latency < 100 ms | Reduce ADC window, enable 240 MHz |
| G4 | Sprint 4 | Real-world accuracy within 8% of clean | Return to Sprint 2 for retraining |
| G5 | Sprint 5 | Paper submitted to target venue | Submit to EMBC conference as backup |

---

## Risk Flags

- [ ] **SRAM overflow**: 260 KB models + OS ~40 KB = ~300 KB, 212 KB headroom — safe
- [ ] **Signal quality**: AD8232 breadboard prototype may have noise — PCB fixes this
- [ ] **Class imbalance**: AFib underrepresented — class weighting + SMOTE planned
- [ ] **Participant recruitment**: Target 10 healthy volunteers — expand to colleagues if needed
- [ ] **Quantization loss**: Calibration set size of 500/class should prevent >5% drop
