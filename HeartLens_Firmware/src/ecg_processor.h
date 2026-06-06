#ifndef ECG_PROCESSOR_H
#define ECG_PROCESSOR_H

#include <cstdint>

struct InferenceResult {
  int classId;       // 0-5
  float confidence;  // 0.0-1.0 — properly dequantized
  bool valid;        // false if inference failed
};

struct WindowResult {
  int classVotes[6];  // accumulated votes across sliding windows
  int totalWindows;
};

class EcgProcessor {
public:
  EcgProcessor();
  bool begin();
  InferenceResult runInference(int16_t* samples, int length);
  WindowResult runSlidingInference(int16_t* samples, int length,
                                    int windowSize, int stride);
  void printModelInfo() const;

private:
  bool _initialized;
  void* _arena;
  int _arenaSize;

  bool loadModel(const unsigned char modelData[], int modelSize,
                 const char* name, void* arena, int arenaSize,
                 int& inputSize, int& outputSize);
  float dequantizeInt8(int8_t value, float scale, int zeroPoint);
};

#endif
