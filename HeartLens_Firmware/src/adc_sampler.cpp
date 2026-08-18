#include "adc_sampler.h"
#include "Config.h"

#include <Arduino.h>
#include <driver/adc.h>
#include <esp_adc_cal.h>

AdcSampler::AdcSampler()
  : _pin(-1), _sampleRate(0), _windowSamples(0),
    _buffer(nullptr), _writeIndex(0), _bufferFull(false),
    _running(false), _taskHandle(nullptr), _mutex(nullptr) {}

bool AdcSampler::begin(int pin, int sampleRate, int windowSamples) {
  _pin = pin;
  _sampleRate = sampleRate;
  _windowSamples = windowSamples;
  _buffer = new int16_t[windowSamples];
  if (!_buffer) return false;
  memset(_buffer, 0, windowSamples * sizeof(int16_t));
  _writeIndex = 0;
  _bufferFull = false;
  _running = false;
  _taskHandle = nullptr;

  _mutex = xSemaphoreCreateMutex();
  if (!_mutex) return false;

#if TARGET_IS_S3
  // S3: ADC1 channels are GPIO1..10 (GPIO4 = ADC1_CH4); the channel
  // number equals the GPIO number for the ADC1 range.
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten((adc1_channel_t)_pin, ADC_CALIB_ATTEN);
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_CALIB_ATTEN,
                            ADC_WIDTH_BIT_12, ADC_CALIB_DEFAULT_VREF,
                            &_adcChars);
#else
  pinMode(_pin, INPUT);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  esp_adc_cal_characterize(ADC_UNIT_1, (adc_atten_t)ADC_CALIB_ATTEN,
                            ADC_WIDTH_BIT_12, ADC_CALIB_DEFAULT_VREF,
                            &_adcChars);
#endif

  return true;
}

void AdcSampler::startSampling() {
  if (_running) return;
  _running = true;
  xTaskCreatePinnedToCore(
    samplingTask, "adc_sampler", 4096, this, 2, &_taskHandle, 0
  );
}

void AdcSampler::stopSampling() {
  if (!_running) return;
  _running = false;
  if (_taskHandle) {
    vTaskDelete(_taskHandle);
    _taskHandle = nullptr;
  }
}

bool AdcSampler::isBufferFull() const {
  return _bufferFull;
}

int16_t* AdcSampler::getWindow() {
  if (_mutex) xSemaphoreTake(_mutex, portMAX_DELAY);
  _bufferFull = false;
  int16_t* buf = _buffer;
  if (_mutex) xSemaphoreGive(_mutex);
  return buf;
}

int AdcSampler::getWindowSize() const {
  return _windowSamples;
}

void AdcSampler::reset() {
  if (_mutex) xSemaphoreTake(_mutex, portMAX_DELAY);
  _writeIndex = 0;
  _bufferFull = false;
  memset(_buffer, 0, _windowSamples * sizeof(int16_t));
  if (_mutex) xSemaphoreGive(_mutex);
}

void AdcSampler::adcReadBlock(int16_t* out, int count) {
  uint32_t raw;
  for (int i = 0; i < count; i++) {
#if TARGET_IS_S3
    raw = adc1_get_raw((adc1_channel_t)_pin);
#else
    raw = analogRead(_pin);
#endif
    raw = esp_adc_cal_raw_to_voltage(raw, &_adcChars);
    out[i] = (int16_t)raw;
  }
}

int AdcSampler::collectReplayWindow(int16_t* out, int outSamples) {
  // Sample at the playback rate (sampleRate * REPLAY_RATE_MULT) with
  // 4x oversampling averaging; 1:1 sample mapping with the DAC output.
  const int ovs = 4;
  for (int i = 0; i < outSamples; i++) {
    int16_t rawBuf[4];
    adcReadBlock(rawBuf, ovs);
    int32_t sum = 0;
    for (int j = 0; j < ovs; j++) sum += rawBuf[j];
    out[i] = (int16_t)(sum / ovs);
  }
  return outSamples;
}

void AdcSampler::samplingTask(void* param) {
  auto* self = static_cast<AdcSampler*>(param);

  const int samplesPerTick = ADC_OVERSAMPLES;
  const TickType_t interval = pdMS_TO_TICKS(1000 / self->_sampleRate);

  while (self->_running) {
    TickType_t lastWake = xTaskGetTickCount();

    if (!self->_bufferFull) {
      int16_t rawBuf[8];
      self->adcReadBlock(rawBuf, samplesPerTick);
      int32_t sum = 0;
      for (int i = 0; i < samplesPerTick; i++) sum += rawBuf[i];
      int16_t sample = (int16_t)(sum / samplesPerTick);

      if (self->_mutex) xSemaphoreTake(self->_mutex, portMAX_DELAY);
      if (!self->_bufferFull) {
        self->_buffer[self->_writeIndex] = sample;
        self->_writeIndex++;
        if (self->_writeIndex >= self->_windowSamples) {
          self->_writeIndex = 0;
          self->_bufferFull = true;
        }
      }
      if (self->_mutex) xSemaphoreGive(self->_mutex);
    }

    vTaskDelayUntil(&lastWake, interval);
  }
  vTaskDelete(NULL);
}
