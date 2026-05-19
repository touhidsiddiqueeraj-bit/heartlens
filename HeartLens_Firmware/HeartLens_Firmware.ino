// HeartLens AI — ESP32 Firmware
// Edge ECG Monitoring with On-Device AI Inference
// State machine: IDLE → SAMPLING → INFERENCE → DISPLAY → IDLE
// ADC on Core 0, Inference + Display on Core 1

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "Config.h"
#include "adc_sampler.h"
#include "ecg_processor.h"
#include "interpreter.h"
#include "display.h"
#include "battery.h"
#include "lead_off.h"
#include "debug.h"

// ─── State Machine ─────────────────────────────────────────────────
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

// ─── Global Instances ──────────────────────────────────────────────
static AdcSampler g_adc;
static EcgProcessor g_ecg;
static Interpreter g_interpreter;
static Display g_display;
static BatteryMonitor g_battery;
static LeadOffDetector g_leadOff;

// ─── State transition helper ───────────────────────────────────────
void setState(DeviceState newState) {
  DEBUG_PRINTF("[State] %d → %d\n", (int)g_state, (int)newState);
  g_state = newState;
  g_stateStart = millis();
}

// ─── Setup ────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== HeartLens AI v1.0 ===");

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  // Initialize modules
  if (!g_display.begin(PIN_OLED_SDA, PIN_OLED_SCL, OLED_I2C_ADDR)) {
    Serial.println("[FATAL] Display init failed");
  } else {
    g_display.showSplash();
  }

  if (!g_ecg.begin()) {
    Serial.println("[FATAL] ECG processor init failed");
    setState(DeviceState::ERROR);
    return;
  }

  if (!g_adc.begin(PIN_ADC_INPUT, SAMPLE_RATE_Hz, SAMPLES_PER_WINDOW)) {
    Serial.println("[FATAL] ADC init failed");
    setState(DeviceState::ERROR);
    return;
  }

  g_battery.begin(PIN_BAT_MONITOR);
  g_leadOff.begin(PIN_LEAD_OFF_P, PIN_LEAD_OFF_N);

  Serial.println("[Setup] All modules initialized");
  Serial.printf("[Setup] Free heap: %d bytes\n", ESP.getFreeHeap());

  delay(2000);
  setState(DeviceState::IDLE);
}

// ─── Main Loop (runs on Core 1) ────────────────────────────────────
void loop() {
  switch (g_state) {

    // ── IDLE ─────────────────────────────────────────────────────
    case DeviceState::IDLE: {
      digitalWrite(PIN_LED, LOW);

      // Check battery first
      if (g_battery.isCritical()) {
        setState(DeviceState::LOW_BATTERY);
        break;
      }

      // Check lead-off
      if (g_leadOff.isDisconnected()) {
        g_display.showIdle();
        delay(500);
        break;
      }

      // Start sampling
      g_adc.reset();
      g_adc.startSampling();
      g_display.showIdle();
      setState(DeviceState::SAMPLING);
      break;
    }

    // ── SAMPLING ─────────────────────────────────────────────────
    case DeviceState::SAMPLING: {
      digitalWrite(PIN_LED, !digitalRead(PIN_LED));  // blink

      // Check lead-off during sampling
      if (g_leadOff.isDisconnected()) {
        g_adc.reset();
        setState(DeviceState::IDLE);
        break;
      }

      // Check battery during sampling (every 5 seconds)
      if ((millis() - g_stateStart) % 5000 < 100) {
        int batPct = g_battery.readPercent();
        g_display.showBattery(batPct);
        DEBUG_PRINTF("[Battery] %d mV (%d%%)\n", g_battery.readMillivolts(), batPct);
      }

      if (g_adc.isBufferFull()) {
        digitalWrite(PIN_LED, HIGH);
        g_adc.stopSampling();
        setState(DeviceState::INFERENCE);
      }
      break;
    }

    // ── INFERENCE ────────────────────────────────────────────────
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

    // ── DISPLAY ──────────────────────────────────────────────────
    case DeviceState::DISPLAY: {
      digitalWrite(PIN_LED, HIGH);

      // Show result for 10 seconds, then return to IDLE
      if (millis() - g_stateStart > 10000) {
        setState(DeviceState::IDLE);
      }

      // Refresh battery bar every 2 seconds
      if ((millis() - g_stateStart) % 2000 < 100) {
        g_display.showBattery(g_battery.readPercent());
      }

      delay(100);
      break;
    }

    // ── LOW_BATTERY ──────────────────────────────────────────────
    case DeviceState::LOW_BATTERY: {
      digitalWrite(PIN_LED, HIGH);
      g_display.showMessage("Battery low. Please charge.", Urgency::Error);
      delay(3000);
      if (!g_battery.isCritical()) {
        setState(DeviceState::IDLE);
      }
      break;
    }

    // ── ERROR ────────────────────────────────────────────────────
    case DeviceState::ERROR: {
      digitalWrite(PIN_LED, HIGH);
      g_display.showMessage("System error. Restart device.", Urgency::Error);
      delay(5000);
      ESP.restart();
      break;
    }

    // ── INIT ─────────────────────────────────────────────────────
    case DeviceState::INIT:
    default:
      break;
  }

  delay(10);  // prevent watchdog reset on Core 1
}
