#ifndef ADC_SAMPLER_H
#define ADC_SAMPLER_H

#include <cstdint>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <esp_adc_cal.h>

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
