#ifndef CONFIG_H
#define CONFIG_H

// ─── Pin Assignments ───────────────────────────────────────────────
#define PIN_ADC_INPUT      34   // AD8232 OUTPUT → ADC1_CH6
#define PIN_LEAD_OFF_P     32   // AD8232 LOFF+
#define PIN_LEAD_OFF_N     33   // AD8232 LOFF-
#define PIN_BAT_MONITOR    35   // Battery divider → ADC1_CH7
#define PIN_OLED_SDA       21   // SSD1306 I2C data
#define PIN_OLED_SCL       22   // SSD1306 I2C clock
#define PIN_LED             2   // Built-in status LED

// ─── ADC Sampling ──────────────────────────────────────────────────
#define SAMPLE_RATE_Hz         360
#define WINDOW_SECONDS         10
#define SAMPLES_PER_WINDOW     (SAMPLE_RATE_Hz * WINDOW_SECONDS)  // 3600
#define ADC_VREF               3.3f
#define ADC_RESOLUTION         4095  // 12-bit

// I2S ADC for jitter-free sampling (replaces analogRead)
#define USE_I2S_ADC            1
#define I2S_ADC_UNIT           ADC1
#define I2S_ADC_CHANNEL        ADC1_CHANNEL_6  // GPIO34

// ADC calibration constants (measure per-device, store in NVS)
#define ADC_CALIB_ATTEN        ADC_ATTEN_DB_11
#define ADC_CALIB_DEFAULT_VREF 1100  // mV — measure actual with esp_adc_cal

// ADC → int8 mapping for TFLite input
#define ADC_VMID              2048
#define ADC_INT8_SHIFT         4
#define INT8_ZERO_POINT        0
#define INT8_SCALE             1.0f

// ─── Inference ─────────────────────────────────────────────────────
#define TENSOR_ARENA_SIZE       (120 * 1024)  // 120 KB shared arena
#define MODEL_INPUT_SAMPLES     360   // 1 second at 360 Hz
#define INFERENCE_STRIDE        180   // 50% overlap sliding window
#define SLIDING_WINDOWS_PER_BUF (SAMPLES_PER_WINDOW / INFERENCE_STRIDE - 1)

// ─── Interpreter Thresholds ────────────────────────────────────────
#define CONFIDENCE_HIGH     0.75f
#define CONFIDENCE_LOW      0.55f
#define NORMAL_DEBOUNCE_MS  30000  // don't re-alert within 30s of normal

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
