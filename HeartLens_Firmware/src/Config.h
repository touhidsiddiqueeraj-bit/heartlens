#ifndef CONFIG_H
#define CONFIG_H

// ─── Board target ──────────────────────────────────────────────────
// ESP32-WROOM-32 (default) or ESP32-S3 (N16R8). Compile-time detection
// via CONFIG_IDF_TARGET_* (set by the toolchain; PlatformIO envs are
// configured for both). S3 differences handled in the #if blocks below.
#if defined(CONFIG_IDF_TARGET_ESP32S3)
  #define TARGET_IS_S3 1
#else
  #define TARGET_IS_S3 0
#endif

// ─── Pin Assignments ───────────────────────────────────────────────
#if TARGET_IS_S3
  // S3 ADC1 channels are GPIO1..10 (GPIO4 = ADC1_CH4). Default ECG input.
  #define PIN_ADC_INPUT      4    // ECG signal → ADC1_CH4
  #define PIN_DAC_OUTPUT     17   // DAC1 — REPLAY_MODE playback output
  #define PIN_LEAD_OFF_P     2    // AD8232 LOFF+ (any GPIO)
  #define PIN_LEAD_OFF_N     3    // AD8232 LOFF- (any GPIO)
  #define PIN_BAT_MONITOR    5    // Battery divider → ADC1_CH4
  #define PIN_OLED_SDA       8    // SSD1306 I2C data
  #define PIN_OLED_SCL       9    // SSD1306 I2C clock
  #define PIN_LED           48    // Built-in status LED (S3 devkit)
#else
  #define PIN_ADC_INPUT      34   // AD8232 OUTPUT → ADC1_CH6
  #define PIN_LEAD_OFF_P     32   // AD8232 LOFF+
  #define PIN_LEAD_OFF_N     33   // AD8232 LOFF-
  #define PIN_BAT_MONITOR    35   // Battery divider → ADC1_CH7
  #define PIN_OLED_SDA       21   // SSD1306 I2C data
  #define PIN_OLED_SCL       22   // SSD1306 I2C clock
  #define PIN_LED             2   // Built-in status LED
#endif

// ─── ADC Sampling ──────────────────────────────────────────────────
#define SAMPLE_RATE_Hz         360
#define WINDOW_SECONDS         10
#define SAMPLES_PER_WINDOW     (SAMPLE_RATE_Hz * WINDOW_SECONDS)  // 3600
#define ADC_VREF               3.3f
#define ADC_RESOLUTION         4095  // 12-bit

// Task-polled adc1_get_raw with 4x oversampling on both targets.
// (ESP32-S3 DMA continuous mode requires ESP-IDF 5.x — upgrade path:
// swap to esp_adc/adc_continuous.h with an IDF 5 toolchain.)
#define ADC_OVERSAMPLES        4

// ADC calibration constants (measure per-device, store in NVS)
#define ADC_CALIB_ATTEN        ADC_ATTEN_DB_11
#define ADC_CALIB_DEFAULT_VREF 1100  // mV — measure actual with esp_adc_cal

// ─── Inference ─────────────────────────────────────────────────────
#define TENSOR_ARENA_SIZE       (120 * 1024)  // 120 KB shared arena
#define MODEL_INPUT_SAMPLES     360   // 1 second at 360 Hz (beat-level)
#define INFERENCE_STRIDE        180   // 50% overlap sliding window
#define SLIDING_WINDOWS_PER_BUF (SAMPLES_PER_WINDOW / INFERENCE_STRIDE - 1)
#define NUM_CLASSES             3     // Normal, APB, PVC (matches training)

// ─── Interpreter Thresholds ────────────────────────────────────────
#define CONFIDENCE_HIGH     0.75f
#define CONFIDENCE_LOW      0.55f
#define NORMAL_DEBOUNCE_MS  30000  // don't re-alert within 30s of normal
// Temperature scaling (fit on validation set by calibrate.py).
// p_cal(i) = p(i)^(1/T) / sum_j p(j)^(1/T). T=1.0 disables calibration.
#define CALIB_TEMPERATURE   1.0f

// ─── Signal Quality ────────────────────────────────────────────────
#define SQI_FLAT_FRACTION   0.05f  // >5% samples saturated → poor
#define SQI_NOISE_THRESHOLD 0.35f  // rms(diff)/rms(signal) above → poor

// ─── Experimental modes ────────────────────────────────────────────
// REPLAY_MODE: PC streams ECG windows over serial → DAC playback at
// REPLAY_RATE_MULT x real time → (jumper GPIO17→GPIO4) → ADC → inference
// → serial dump. Hardware-domain-shift experiment (Exp 6). S3 only.
#define REPLAY_MODE         0
#define REPLAY_RATE_MULT    10     // playback/ADC speed-up factor
// BENCHMARK_MODE: prints per-stage micros timing report to serial on
// demand (send 'r' in the monitor). Exp 5.
#define BENCHMARK_MODE      0

// ─── Battery ───────────────────────────────────────────────────────
#define BAT_WARN_MV         3400
#define BAT_STOP_MV         3200
#define BAT_DIVIDER_R1      10.0f  // kOhm
#define BAT_DIVIDER_R2      10.0f  // kOhm
#define BAT_ADC_SAMPLES     16     // averaging filter

// ─── Lead-off ──────────────────────────────────────────────────────
#define LEAD_OFF_THRESHOLD_MS   200   // ms before declaring lead-off
#define LEAD_OFF_RECONNECT_MS   500   // ms hysteresis on reconnect

// ─── Display ───────────────────────────────────────────────────────
#define OLED_I2C_ADDR       0x3C
#define OLED_I2C_RETRIES    3
#define OLED_I2C_TIMEOUT_MS 100

// ─── Watchdog ──────────────────────────────────────────────────────
#define WDT_TIMEOUT_MS      10000   // 10 second hardware watchdog
#define WDT_FEED_INTERVAL_MS 2000  // feed every 2s in main loop

// ─── Debug ─────────────────────────────────────────────────────────
// #define DEBUG_ENABLED

#endif // CONFIG_H
