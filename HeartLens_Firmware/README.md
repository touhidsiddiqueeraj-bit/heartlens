# HeartLens AI — ESP32 Firmware

Offline ECG screening device with on-device AI inference. Runs entirely on an ESP32 DevKit v4 — no smartphone, no cloud, no internet required.

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
# Convert trained .tflite → C header arrays
./convert_tflite_to_headers.sh ../heart-lens-training/models models/
```

---

## Hardware Pin Mapping

| ESP32 Pin | Connects To | Function |
|-----------|-------------|----------|
| GPIO34 | AD8232 OUTPUT | ECG signal (ADC1_CH6) |
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
Electrode RA ──→ AD8232 IN-     AD8232 OUTPUT ──→ ESP32 GPIO34 (ADC)
Electrode RL ──→ AD8232 RLD     LOFF+ ──→ GPIO32, LOFF- ──→ GPIO33
```

---

## State Machine

```
         ┌──────────────────────────────────────┐
         │                                      ▼
    ┌────────┐  lead_off   ┌──────────┐  buffer  ┌───────────┐
    │  IDLE  │───────────▶│ SAMPLING │───full──▶│ INFERENCE │
    └────┬───┘            └──────────┘          └─────┬─────┘
         ▲                                            │
         │                 ┌──────────┐                │
         │                 │ DISPLAY  │◄───────────────┘
         │                 └────┬─────┘
         │                      │ 10s timeout
         └──────────────────────┘
```

| State | Action | Duration |
|-------|--------|----------|
| **IDLE** | Check battery, check lead-off, start ADC | Until start trigger |
| **SAMPLING** | ADC fill circular buffer (360 Hz, 3600 samples) | 10 seconds |
| **INFERENCE** | Run denoiser + classifier + interpreter | ~50-100 ms |
| **DISPLAY** | Show result, refresh battery bar | 10 seconds |
| **LOW_BATTERY** | Show warning, halt inference | Until charged >3.2V |
| **ERROR** | Show error message, restart | 5 seconds |

---

## File Structure

```
HeartLens_Firmware/
├── HeartLens_Firmware.ino       # Main: setup + loop + state machine
├── src/
│   ├── Config.h                 # All pins, thresholds, constants
│   ├── adc_sampler.h/.cpp       # ADC sampling on Core 0 (FreeRTOS)
│   ├── ecg_processor.h/.cpp     # TFLite Micro inference pipeline
│   ├── interpreter.h/.cpp       # Confidence → plain-language output
│   ├── display.h/.cpp           # SSD1306 OLED driver
│   ├── battery.h/.cpp           # Battery voltage monitoring
│   ├── lead_off.h/.cpp          # Electrode disconnect detection
│   └── debug.h                  # Serial debug macros
├── models/
│   ├── denoiser_model.h         # int8 quantized denoiser (C array)
│   └── classifier_model.h       # int8 quantized classifier (C array)
├── lib/tensorflow_lite/         # TFLite Micro source (25 MB)
├── setup_tflite_micro.sh        # Download script for TFLite Micro
└── convert_tflite_to_headers.sh # .tflite → .h conversion script
```

---

## Firmware Architecture

### ADC Sampling (Core 0)

A FreeRTOS task pinned to Core 0 reads GPIO34 at 360 Hz using `analogRead()`. Samples are stored in a 3600-element circular buffer (`int16_t`). When the buffer wraps (10 seconds elapsed), a flag signals the main loop on Core 1.

### TFLite Inference Pipeline (Core 1)

Two int8 quantized models run sequentially in a shared 120 KB arena:

1. **Denoiser** (Conv1D autoencoder): input [1,360,1] → output [1,360,1]
   - Removes motion artifacts, baseline wander, PLI, EMG noise
2. **Classifier** (1D-CNN): input [1,360,1] → output [1,6]
   - 3 conv blocks (8/16/32 filters), global average pooling, softmax

Raw ADC values (0-4095) are mapped to int8 (-128 to 127) before inference.

### Rule Interpreter

| Confidence | Output |
|------------|--------|
| > 0.75 | Class-specific message (Normal / Seek attention / See doctor) |
| 0.55 – 0.75 | "Signal unclear. Please reattach electrodes and try again." |
| < 0.55 | "Signal unclear. Please reattach electrodes and try again." |

### Display Output (SSD1306 128x64)

Four possible plain-language messages:
1. "Heart rhythm looks normal."
2. "Unusual rhythm detected. Please see a doctor soon."
3. "Irregular rhythm detected. Please seek medical attention."
4. "Signal unclear. Please reattach electrodes and try again."

A battery icon in the top-right corner shows charge level. Status bar at top shows urgency (NORMAL / CAUTION / ALERT / ERROR).

---

## Configuration (`src/Config.h`)

| Setting | Default | Description |
|---------|---------|-------------|
| SAMPLE_RATE_Hz | 360 | ADC sampling frequency |
| WINDOW_SECONDS | 10 | Buffer duration for one inference |
| TENSOR_ARENA_SIZE | 122880 (120 KB) | Shared TFLite memory arena |
| CONFIDENCE_HIGH | 0.75 | Threshold for confident result |
| CONFIDENCE_LOW | 0.55 | Threshold for unclear signal |
| BAT_WARN_MV | 3400 | Low battery warning threshold |
| BAT_STOP_MV | 3200 | Critical battery halt threshold |

---

## Debugging

Enable debug output by uncommenting in `src/debug.h`:

```cpp
#define DEBUG_ENABLED
```

This enables Serial output showing:
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
python3 train_denoiser.py           # Train LSTM denoiser → int8 TFLite
python3 train_classifier.py          # Train 1D-CNN classifier → int8 TFLite
```

See `../heart-lens-training/TRAINING_DATA.md` for dataset statistics.

---

## License

MIT — see LICENSE file in project root.
