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

#if USE_I2S_ADC
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(I2S_ADC_CHANNEL, ADC_CALIB_ATTEN);
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_CALIB_ATTEN,
                            ADC_WIDTH_BIT_12, ADC_CALIB_DEFAULT_VREF,
                            &_adcChars);
#else
  pinMode(_pin, INPUT);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_11db,
                            ADC_WIDTH_BIT_12, ADC_CALIB_DEFAULT_VREF,
                            &_adcChars);
#endif

  return true;
}

void AdcSampler::startSampling() {
  if (_running) return;
  _running = true;
  xTaskCreatePinnedToCore(
    samplingTask, "adc_sampler", 3072, this, 2, &_taskHandle, 0
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
#if USE_I2S_ADC
    raw = adc1_get_raw(I2S_ADC_CHANNEL);
#else
    raw = analogRead(_pin);
#endif
    raw = esp_adc_cal_raw_to_voltage(raw, &_adcChars);
    out[i] = (int16_t)raw;
  }
}

void AdcSampler::samplingTask(void* param) {
  auto* self = static_cast<AdcSampler*>(param);

  // Use I2S ADC for precise periodic sampling
  const int samplesPerTick = 4;  // oversample ×4, average
  const TickType_t interval = pdMS_TO_TICKS(1000 / self->_sampleRate);

  while (self->_running) {
    TickType_t lastWake = xTaskGetTickCount();

    if (!self->_bufferFull) {
      int16_t rawBuf[samplesPerTick];
      int32_t sum = 0;
      self->adcReadBlock(rawBuf, samplesPerTick);
      for (int i = 0; i < samplesPerTick; i++) sum += rawBuf[i];
      int16_t sample = sum / samplesPerTick;

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
