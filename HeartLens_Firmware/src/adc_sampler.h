#ifndef ADC_SAMPLER_H
#define ADC_SAMPLER_H

#include <cstdint>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include "Config.h"
#include <esp_adc_cal.h>

// NOTE: The ESP32-S3 continuous-mode (DMA) driver requires ESP-IDF 5.x.
// This toolchain (Arduino core 3.x / IDF 4.4) uses the legacy calibrated
// adc1_* API on both targets; the sampling task keeps a fixed cadence,
// so jitter is bounded regardless of driver. Upgrade path: swap to
// esp_adc/adc_continuous.h when moving to an IDF 5.x toolchain.

class AdcSampler {
public:
  AdcSampler();
  bool begin(int pin, int sampleRate, int windowSamples);
  void startSampling();
  void stopSampling();
  bool isBufferFull() const;
  int16_t* getWindow();
  int getWindowSize() const;
  void reset();

  // REPLAY_MODE: capture `outSamples` at the playback rate (equal to
  // sampleRate * REPLAY_RATE_MULT) with 4x oversampling averaging.
  // Returns number of output samples written.
  int collectReplayWindow(int16_t* out, int outSamples);

private:
  int _pin;
  int _sampleRate;
  int _windowSamples;
  int16_t* _buffer;
  volatile int _writeIndex;
  volatile bool _bufferFull;
  volatile bool _running;
  TaskHandle_t _taskHandle;
  SemaphoreHandle_t _mutex;

  esp_adc_cal_characteristics_t _adcChars;

  static void samplingTask(void* param);
  void adcReadBlock(int16_t* out, int count);
};

#endif
