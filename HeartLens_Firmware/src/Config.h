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
#define SAMPLE_RATE_Hz     360
#define WINDOW_SECONDS      10
#define SAMPLES_PER_WINDOW (SAMPLE_RATE_Hz * WINDOW_SECONDS)  // 3600
#define ADC_VREF            3.3
#define ADC_RESOLUTION      4095  // 12-bit
#define I2S_SAMPLE_RATE     (SAMPLE_RATE_Hz * 4)  // I2S oversample factor

// ─── TFLite ────────────────────────────────────────────────────────
#define TENSOR_ARENA_SIZE   (120 * 1024)  // 120 KB shared arena

// ─── Interpreter Thresholds ────────────────────────────────────────
#define CONFIDENCE_HIGH     0.75f
#define CONFIDENCE_LOW      0.55f

// ─── Battery ───────────────────────────────────────────────────────
#define BAT_WARN_MV         3400
#define BAT_STOP_MV         3200
#define BAT_DIVIDER_R1      10.0f  // kOhm
#define BAT_DIVIDER_R2      10.0f  // kOhm

// ─── Lead-Off ──────────────────────────────────────────────────────
#define LEAD_OFF_THRESHOLD_MS 200  // ms before declaring lead-off

// ─── Display ───────────────────────────────────────────────────────
#define OLED_I2C_ADDR       0x3C

// ─── Debug ─────────────────────────────────────────────────────────
// #define DEBUG_ENABLED

#endif // CONFIG_H
