#include "adc_sampler.h"
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

AdcSampler::AdcSampler()
  : _pin(-1), _sampleRate(0), _windowSamples(0),
    _buffer(nullptr), _writeIndex(0), _bufferFull(false) {}

bool AdcSampler::begin(int pin, int sampleRate, int windowSamples) {
  _pin = pin;
  _sampleRate = sampleRate;
  _windowSamples = windowSamples;
  _buffer = new int16_t[windowSamples];
  if (!_buffer) return false;
  memset(_buffer, 0, windowSamples * sizeof(int16_t));
  _writeIndex = 0;
  _bufferFull = false;
  pinMode(_pin, INPUT);
  analogReadResolution(12);
  return true;
}

void AdcSampler::startSampling() {
  xTaskCreatePinnedToCore(
    samplingTask, "adc_sampler", 2048, this, 2, NULL, 0
  );
}

void AdcSampler::stopSampling() {
  // Task deletion handled externally
}

bool AdcSampler::isBufferFull() const {
  return _bufferFull;
}

int16_t* AdcSampler::getWindow() {
  _bufferFull = false;
  return _buffer;
}

int AdcSampler::getWindowSize() const {
  return _windowSamples;
}

void AdcSampler::reset() {
  _writeIndex = 0;
  _bufferFull = false;
  memset(_buffer, 0, _windowSamples * sizeof(int16_t));
}

void AdcSampler::samplingTask(void* param) {
  auto* self = static_cast<AdcSampler*>(param);
  const TickType_t interval = pdMS_TO_TICKS(1000 / self->_sampleRate);

  while (true) {
    TickType_t lastWake = xTaskGetTickCount();
    if (!self->_bufferFull) {
      self->_buffer[self->_writeIndex] = analogRead(self->_pin);
      self->_writeIndex++;
      if (self->_writeIndex >= self->_windowSamples) {
        self->_writeIndex = 0;
        self->_bufferFull = true;
      }
    }
    vTaskDelayUntil(&lastWake, interval);
  }
}
