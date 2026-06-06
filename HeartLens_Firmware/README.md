# HeartLens AI — ESP32 Firmware v1.1

Offline ECG screening device with on-device AI inference. Runs entirely on an ESP32 DevKit v4 — no smartphone, no cloud, no internet required.

## What's new in v1.1

| Feature | v1.0 | v1.1 |
|---------|------|------|
| ADC method | `analogRead()` (blocking, jittery) | I2S-driven with `esp_adc_cal` |
| Thread safety | `volatile` only (data race) | FreeRTOS mutex |
| ADC task lifecycle | Never stops (writes during inference) | Proper `vTaskDelete` |
| Denoiser | LSTM autoencoder (~148 KB, slow) | Conv1D autoencoder (~50 KB, 3-5x faster) |
| Window utilization | 1s analyzed from 10s buffer | Sliding window (50% overlap, majority vote) |
| Dequantization | `(s+128)/255` (wrong) | scale/zero_point from TFLite tensor |
| Interpreter | Both branches return same message | Working LOW < HIGH + debounce |
| Battery | Single read, linear voltage | 16-sample avg + LiPo lookup table |
| Lead-off | No reconnect hysteresis | 500ms hysteresis |
| Display | No I2C error recovery | Retry on init, I2C ping before write |
| Watchdog | `delay(10)` hack | `esp_task_wdt` configured explicitly |

## Quick Start

### Prerequisites

- **Arduino IDE** with **ESP32 board package** installed
- Libraries (install via Library Manager):
  - `Adafruit SSD1306`
  - `Adafruit GFX`
- TFLite Micro source (run setup script below)

### Setup

```bash
# 1. Download TFLite Micro source into lib/tensorflow_lite
./setup_tflite_micro.sh

# 2. Open HeartLens_Firmware.ino in Arduino IDE
# 3. Select Board: "ESP32 Dev Module"
# 4. Select Port: /dev/ttyUSB0 (or your port)
# 5. Click Upload
```

### After Training Real Models

```bash
# Convert trained .tflite → C header arrays (with extern "C" guards)
./convert_tflite_to_headers.sh ../heart-lens-training/models models/
```

---

## Hardware Pin Mapping

| ESP32 Pin | Connects To | Function |
|-----------|-------------|----------|
| GPIO34 | AD8232 OUTPUT | ECG signal (ADC1_CH6, I2S-driven) |
| GPIO32 | AD8232 LOFF+ | Lead-off detect positive |
| GPIO33 | AD8232 LOFF- | Lead-off detect negative |
| GPIO21 | SSD1306 SDA | I2C data (4.7k pull-up) |
| GPIO22 | SSD1306 SCL | I2C clock (4.7k pull-up) |
| GPIO35 | Battery divider | LiPo voltage monitor (ADC1_CH7) |
| GPIO2 | Built-in LED | Status indicator |
| GPIO1 | UART TX | Debug serial output |
| GPIO3 | UART RX | Debug serial input |

### Power Connections

| Component | Connects To |
|-----------|-------------|
| USB-C VBUS | TP4056 VIN |
| TP4056 VOUT+ | LiPo BAT+ |
| LiPo BAT+ | AMS1117-3.3 VIN |
| AMS1117-3.3 VOUT | 3.3V rail (ESP32, AD8232, SSD1306) |
| LiPo BAT+ → 10kΩ → GPIO35 → 10kΩ → GND | Battery voltage divider |

### Analog Signal Path

```
Electrode LA ──→ AD8232 IN+
Electrode RA ──→ AD8232 IN-     AD8232 OUTPUT ──→ ESP32 GPIO34 (I2S ADC)
Electrode RL ──→ AD8232 RLD     LOFF+ ──→ GPIO32, LOFF- ──→ GPIO33
```

---

## State Machine

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

| State | Action | Duration |
|-------|--------|----------|
| **IDLE** | Check battery, check lead-off, start ADC | Until start trigger |
| **SAMPLING** | ADC fill circular buffer (I2S, 360 Hz, 3600 samples) | 10 seconds |
| **INFERENCE** | Conv1D denoiser + 1D-CNN classifier (sliding window) | ~50-100 ms |
| **DISPLAY** | Show result, refresh battery bar | 10 seconds |
| **LOW_BATTERY** | Show warning, halt inference | Until charged >3.2V |
| **ERROR** | Show error message, restart | 5 seconds |

---

## File Structure

```
HeartLens_Firmware/
├── HeartLens_Firmware.ino       # Main: setup + loop + state machine (v1.1)
├── src/
│   ├── Config.h                 # All pins, thresholds, ADC cal, WDT config
│   ├── adc_sampler.h/.cpp       # I2S ADC sampling on Core 0 (FreeRTOS + mutex)
│   ├── ecg_processor.h/.cpp     # TFLite Micro inference (sliding window)
│   ├── interpreter.h/.cpp       # Confidence → plain-language output
│   ├── display.h/.cpp           # SSD1306 OLED (I2C error recovery)
│   ├── battery.h/.cpp           # LiPo monitoring (16-sample avg + lookup)
│   ├── lead_off.h/.cpp          # Electrode disconnect (hysteresis)
│   └── debug.h                  # Serial debug macros
├── models/
│   ├── denoiser_model.h         # Conv1D int8 quantized denoiser (C array)
│   └── classifier_model.h       # int8 quantized classifier (C array)
├── lib/tensorflow_lite/         # TFLite Micro source (25 MB)
├── setup_tflite_micro.sh        # Download script for TFLite Micro
└── convert_tflite_to_headers.sh # .tflite → .h with extern "C" guards
```

---

## Firmware Architecture

### ADC Sampling (Core 0)

A FreeRTOS task pinned to Core 0 reads GPIO34 at 360 Hz using **I2S-driven ADC** with `esp_adc_cal` for voltage calibration. Samples are 4x oversampled and averaged per tick. A FreeRTOS **mutex** protects the shared buffer between cores. The task is properly destroyed during inference phase.

### TFLite Inference Pipeline (Core 1)

Two int8 quantized models run with **sliding window inference** (50% overlap):

1. **Denoiser** (Conv1D autoencoder): input [1,360,1] → output [1,360,1]
   - 3-5x faster than LSTM on ESP32
   - Removes motion artifacts, baseline wander, PLI, EMG noise
2. **Classifier** (1D-CNN): input [1,360,1] → output [1,6]
   - 3 conv blocks (32/64/128 filters), sliding window + majority vote

Raw ADC values (millivolts) are centered at 1650mV and scaled to int8.

### Rule Interpreter

| Confidence Level | Behavior |
|-----------------|----------|
| ≥ 0.75 | Class-specific message with urgency |
| 0.55 – 0.75 | Class-specific message with "Possible:" prefix |
| < 0.55 | "Signal unclear" message |
| Normal within 30s debounce | Normal result shown regardless of current reading |

### Display Output (SSD1306 128x64)

Four plain-language messages with urgency color bar. Battery icon with lookup-table-based percentage. I2C communication is verified before each write with automatic recovery.

---

## Configuration (`src/Config.h`)

| Setting | Default | Description |
|---------|---------|-------------|
| SAMPLE_RATE_Hz | 360 | ADC sampling frequency |
| WINDOW_SECONDS | 10 | Buffer duration for one inference |
| TENSOR_ARENA_SIZE | 122880 (120 KB) | Shared TFLite memory arena |
| INFERENCE_STRIDE | 180 | Sliding window stride (50% overlap) |
| CONFIDENCE_HIGH | 0.75 | Threshold for confident result |
| CONFIDENCE_LOW | 0.55 | Threshold for unclear signal |
| NORMAL_DEBOUNCE_MS | 30000 | Suppress alerts after normal reading |
| BAT_WARN_MV | 3400 | Low battery warning threshold |
| BAT_STOP_MV | 3200 | Critical battery halt threshold |
| BAT_ADC_SAMPLES | 16 | Battery averaging filter |
| WDT_TIMEOUT_MS | 10000 | Hardware watchdog timeout |

---

## Debugging

Enable debug output by uncommenting in `src/debug.h`:

```cpp
#define DEBUG_ENABLED
```

Output includes:
- State transitions
- Per-stage inference timestamps (micros)
- Class scores and confidence
- Free heap memory
- Battery voltage and percent

Connect UART at 115200 baud.

---

## Energy Budget

| Component | Current |
|-----------|---------|
| ESP32 (active, 240 MHz) | ~60 mA |
| AD8232 | ~0.5 mA |
| SSD1306 (average) | ~10 mA |
| AMS1117 quiescent | ~5 mA |
| **Total** | **~75 mA** |
| **Battery life (500 mAh)** | **~6.5 hours** |

---

## Model Training

Training scripts live in `../heart-lens-training/`:

```bash
cd ../heart-lens-training
python3 data_loader.py              # Download MIT-BIH, extract segments
python3 train_denoiser.py           # Train Conv1D denoiser → int8 TFLite
python3 train_classifier.py         # Train 1D-CNN classifier → int8 TFLite
```

See `../heart-lens-training/TRAINING_DATA.md` for dataset statistics.

---

## License

MIT — see LICENSE file in project root.
