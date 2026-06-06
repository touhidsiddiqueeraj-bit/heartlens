// HeartLens AI — ESP32 Firmware
// Edge ECG Monitoring with On-Device AI Inference
// State machine: IDLE -> SAMPLING -> INFERENCE -> DISPLAY -> IDLE
// ADC on Core 0, Inference + Display on Core 1

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_task_wdt.h>

#include "Config.h"
#include "adc_sampler.h"
#include "ecg_processor.h"
#include "interpreter.h"
#include "display.h"
#include "battery.h"
#include "lead_off.h"
#include "debug.h"

// State Machine
enum class DeviceState {
  INIT,
  IDLE,
  SAMPLING,
  INFERENCE,
  DISPLAY,
  LOW_BATTERY,
  ERROR
};

static DeviceState g_state = DeviceState::INIT;
static unsigned long g_stateStart = 0;

// Global Instances
static AdcSampler g_adc;
static EcgProcessor g_ecg;
static Interpreter g_interpreter;
static Display g_display;
static BatteryMonitor g_battery;
static LeadOffDetector g_leadOff;

static unsigned long g_lastWdtFeed = 0;
static bool g_fatalError = false;

void setState(DeviceState newState) {
  DEBUG_PRINTF("[State] %d -> %d\n", (int)g_state, (int)newState);
  g_state = newState;
  g_stateStart = millis();
}

void feedWatchdog() {
  unsigned long now = millis();
  if (now - g_lastWdtFeed > WDT_FEED_INTERVAL_MS) {
    esp_task_wdt_reset();
    g_lastWdtFeed = now;
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== HeartLens AI v1.1 ===");

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  // Initialize watchdog
  esp_task_wdt_init(WDT_TIMEOUT_MS / 1000, true);
  esp_task_wdt_add(NULL);

  if (!g_display.begin(PIN_OLED_SDA, PIN_OLED_SCL, OLED_I2C_ADDR)) {
    Serial.println("[WARN] Display init failed — continuing without display");
  } else {
    g_display.showSplash();
  }

  if (!g_ecg.begin()) {
    Serial.println("[FATAL] ECG processor init failed");
    g_fatalError = true;
    setState(DeviceState::ERROR);
    return;
  }

  if (!g_adc.begin(PIN_ADC_INPUT, SAMPLE_RATE_Hz, SAMPLES_PER_WINDOW)) {
    Serial.println("[FATAL] ADC init failed");
    g_fatalError = true;
    setState(DeviceState::ERROR);
    return;
  }

  g_battery.begin(PIN_BAT_MONITOR);
  g_leadOff.begin(PIN_LEAD_OFF_P, PIN_LEAD_OFF_N);

  Serial.println("[Setup] All modules initialized");
  Serial.printf("[Setup] Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("[Setup] CPU freq: %d MHz\n", ESP.getCpuFreqMHz());

  delay(2000);
  setState(DeviceState::IDLE);
}

void loop() {
  feedWatchdog();
  if (g_fatalError) {
    delay(100);
    return;
  }

  switch (g_state) {

    case DeviceState::IDLE: {
      digitalWrite(PIN_LED, LOW);

      if (g_battery.isCritical()) {
        setState(DeviceState::LOW_BATTERY);
        break;
      }

      if (g_leadOff.isDisconnected()) {
        g_display.showIdle();
        delay(500);
        break;
      }

      g_adc.reset();
      g_adc.startSampling();
      g_display.showIdle();
      setState(DeviceState::SAMPLING);
      break;
    }

    case DeviceState::SAMPLING: {
      digitalWrite(PIN_LED, !digitalRead(PIN_LED));

      if (g_leadOff.isDisconnected()) {
        g_adc.stopSampling();
        g_adc.reset();
        setState(DeviceState::IDLE);
        break;
      }

      if (g_adc.isBufferFull()) {
        digitalWrite(PIN_LED, HIGH);
        g_adc.stopSampling();
        setState(DeviceState::INFERENCE);
      }
      break;
    }

    case DeviceState::INFERENCE: {
      digitalWrite(PIN_LED, HIGH);

      int16_t* window = g_adc.getWindow();
      int windowSize = g_adc.getWindowSize();

      InferenceResult result = g_ecg.runInference(window, windowSize);

      OutputMessage msg = g_interpreter.interpret(result);
      g_display.showMessage(msg.message, msg.urgency);

      setState(DeviceState::DISPLAY);
      break;
    }

    case DeviceState::DISPLAY: {
      digitalWrite(PIN_LED, HIGH);

      if (millis() - g_stateStart > 10000) {
        setState(DeviceState::IDLE);
      }

      if ((millis() - g_stateStart) % 2000 < 100) {
        int batPct = g_battery.readPercent();
        g_display.showBattery(batPct);
      }

      delay(100);
      break;
    }

    case DeviceState::LOW_BATTERY: {
      digitalWrite(PIN_LED, HIGH);
      g_display.showMessage("Battery low. Please charge.", Urgency::Error);
      delay(3000);
      if (!g_battery.isCritical()) {
        setState(DeviceState::IDLE);
      }
      break;
    }

    case DeviceState::ERROR: {
      digitalWrite(PIN_LED, HIGH);
      g_display.showMessage("System error. Restart device.", Urgency::Error);
      delay(5000);
      ESP.restart();
      break;
    }

    case DeviceState::INIT:
    default:
      break;
  }

  delay(10);
}
