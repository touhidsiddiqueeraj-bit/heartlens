// HeartLens AI — ESP32 Firmware
// Edge ECG Monitoring with On-Device AI Inference
// State machine: IDLE -> SAMPLING -> INFERENCE -> DISPLAY -> IDLE
// ADC on Core 0, Inference + Display on Core 1
// Experimental modes: REPLAY_MODE (DAC playback loopback), BENCHMARK_MODE

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
  SHOW_RESULT,
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

// ─── REPLAY_MODE: DAC playback of serial-streamed ECG ──────────────
// PC sends 16-bit count + count raw bytes (0-255, 8-bit DAC values).
// The buffer is played back on the S3 DAC at SAMPLE_RATE*REPLAY_RATE_MULT
// Hz while the ADC samples concurrently; the decimated window then runs
// through the full inference pipeline and the weighted result is printed
// as "HLR_RES <class> <conf>" for the host to parse. Requires a jumper
// from PIN_DAC_OUTPUT to PIN_ADC_INPUT (loopback) — hardware replay
// without an analog frontend (Exp 6.2).
#if REPLAY_MODE && TARGET_IS_S3
#include <driver/dac.h>
#include <driver/timer.h>

static volatile int g_replayPos = -1;
static int g_replayLen = 0;
static uint8_t* g_replayBuf = nullptr;

static void IRAM_ATTR replayTimerIsr() {
  if (g_replayPos >= 0 && g_replayPos < g_replayLen) {
    dac_output_voltage(PIN_DAC_OUTPUT, g_replayBuf[g_replayPos]);
    g_replayPos++;
  }
}

static void startReplayTimer(int sampleRate) {
  // 80 MHz APB; timer0
  timer_config_t cfg = {};
  cfg.divider = 80;
  cfg.counter_dir = TIMER_COUNT_UP;
  cfg.auto_reload = true;
  timer_init(TIMER_GROUP_0, TIMER_0, &cfg);
  timer_set_counter_value(TIMER_GROUP_0, TIMER_0, 0);
  timer_set_alarm_value(TIMER_GROUP_0, TIMER_0, 1000000 / sampleRate);
  timer_enable_alarm(TIMER_GROUP_0, TIMER_0, true);
  timer_isr_callback_add(TIMER_GROUP_0, TIMER_0, replayTimerIsr, nullptr);
  timer_start(TIMER_GROUP_0, TIMER_0);
}

static void stopReplayTimer() {
  timer_stop(TIMER_GROUP_0, TIMER_0);
  timer_isr_callback_remove(TIMER_GROUP_0, TIMER_0);
  g_replayPos = -1;
}

static void replayLoop() {
  feedWatchdog();
  if (!g_replayBuf) {
    g_replayBuf = (uint8_t*)malloc(SAMPLES_PER_WINDOW);
    if (!g_replayBuf) {
      Serial.println("[REPLAY] alloc failed");
      delay(5000);
      ESP.restart();
    }
  }

  // Wait for frame header: 'H' 'L' 'R' + uint16 count
  while (Serial.available() < 5) { feedWatchdog(); delay(1); }
  if (Serial.read() != 'H' || Serial.read() != 'L' || Serial.read() != 'R') {
    while (Serial.available()) Serial.read();
    return;
  }
  uint16_t count = (uint16_t)(Serial.read()) | ((uint16_t)Serial.read() << 8);
  if (count > SAMPLES_PER_WINDOW) count = SAMPLES_PER_WINDOW;

  while (Serial.available() < count) { feedWatchdog(); delay(1); }
  for (int i = 0; i < count; i++) g_replayBuf[i] = (uint8_t)Serial.read();

  // Play the window at REPLAY_RATE_MULT x while ADC captures concurrently
  static int16_t replayWindow[SAMPLES_PER_WINDOW];
  g_replayLen = count;
  g_replayPos = 0;
  int playbackRate = SAMPLE_RATE_Hz * REPLAY_RATE_MULT;
  startReplayTimer(playbackRate);

  int got = g_adc.collectReplayWindow(replayWindow, count);

  while (g_replayPos < g_replayLen) { feedWatchdog(); delay(1); }  // finish playback
  stopReplayTimer();

  if (got != count) {
    Serial.println("HLR_ERR");
    return;
  }

  InferenceResult result = g_ecg.runInference(replayWindow, got);
  if (!result.valid) {
    Serial.println(result.signalOk ? "HLR_ERR" : "HLR_UNCLR");
    return;
  }
  Serial.printf("HLR_RES %d %.3f\n", result.classId, result.confidence);
}
#endif  // REPLAY_MODE && TARGET_IS_S3

// ─── BENCHMARK_MODE: synthetic-window latency measurement ──────────
#if BENCHMARK_MODE
static unsigned long g_lastBenchBeat = 0;
static void benchmarkLoop() {
  feedWatchdog();
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') {
      static int16_t synth[SAMPLES_PER_WINDOW];
      // Synthetic "ECG": 1.2 Hz sawtooth + noise, enough to exercise
      // the full pipeline without a real signal
      unsigned long seed = millis();
      for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
        seed = seed * 1103515245UL + 12345UL;
        float phase = (i % 300) / 300.0f * 6.283f;
        synth[i] = (int16_t)(2048 + (int)(700.0f * sinf(phase)) + (int)((seed >> 16) % 40));
      }
      g_ecg.benchDenoiseUs = 0;
      g_ecg.benchClassifyUs = 0;
      g_ecg.benchTotalUs = 0;
      g_ecg.benchWindows = 0;
      unsigned long t0 = micros();
      // Benchmark the NN pipeline directly (window denoise -> classify).
      // Bypasses the signal-quality gate, which is not part of inference
      // and would reject a synthetic buffer (flat/extreme saturation).
      WindowResult wr = g_ecg.runSlidingInference(synth, SAMPLES_PER_WINDOW,
                                                  MODEL_INPUT_SAMPLES, INFERENCE_STRIDE);
      unsigned long tTotal = micros() - t0;
      int best = 0; float bestS = -1.0f, tot = 0.0f;
      for (int i = 0; i < 8; i++) {
        tot += wr.classScores[i];
        if (wr.classScores[i] > bestS) { bestS = wr.classScores[i]; best = i; }
      }
      Serial.printf("BENCH total=%lu us  windows=%lu  denoise_avg=%lu us  classify_avg=%lu us  per_window_avg=%lu us\n",
                    tTotal, g_ecg.benchWindows,
                    g_ecg.benchWindows ? g_ecg.benchDenoiseUs / g_ecg.benchWindows : 0,
                    g_ecg.benchWindows ? g_ecg.benchClassifyUs / g_ecg.benchWindows : 0,
                    g_ecg.benchWindows ? g_ecg.benchTotalUs / g_ecg.benchWindows : 0);
      Serial.printf("  result: class=%d conf=%.3f valid=%d windows=%d\n",
                    best, (tot > 0.0f) ? bestS / tot : 0.0f, wr.totalWindows > 0, wr.totalWindows);
    }
  } else if (millis() - g_lastBenchBeat >= 2000) {
    g_lastBenchBeat = millis();
    Serial.println("bench-wait");  // heartbeat: proves serial path is alive
  }
  delay(10);
}
#endif  // BENCHMARK_MODE

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== HeartLens AI v1.2 ===");
  Serial.println("BOOTP1");

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

#if REPLAY_MODE && TARGET_IS_S3
  dac_output_enable(PIN_DAC_OUTPUT);
  Serial.println("[Setup] REPLAY_MODE active — streaming ECG over serial");
#elif BENCHMARK_MODE
  Serial.println("[Setup] BENCHMARK_MODE active — send 'r' to run benchmark");
#else
  g_battery.begin(PIN_BAT_MONITOR);
  g_leadOff.begin(PIN_LEAD_OFF_P, PIN_LEAD_OFF_N);
#endif

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

#if REPLAY_MODE && TARGET_IS_S3
  replayLoop();
  return;
#elif BENCHMARK_MODE
  benchmarkLoop();
  return;
#endif

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

      setState(DeviceState::SHOW_RESULT);
      break;
    }

    case DeviceState::SHOW_RESULT: {
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
                            