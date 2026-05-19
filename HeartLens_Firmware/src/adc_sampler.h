#ifndef ADC_SAMPLER_H
#define ADC_SAMPLER_H

#include <cstdint>

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

  static void samplingTask(void* param);
};

#endif
