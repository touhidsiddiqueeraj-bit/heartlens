#ifndef ECG_PROCESSOR_H
#define ECG_PROCESSOR_H

#include <cstdint>

struct InferenceResult {
  int classId;       // 0..NUM_CLASSES-1
  float confidence;  // 0.0-1.0 — mean calibrated probability of classId
  float probs[8];    // normalized mean class distribution (weighted vote)
  bool valid;        // false if inference failed
  bool signalOk;     // false if signal-quality gate rejected the buffer
};

struct WindowResult {
  float classScores[8];  // summed softmax probabilities (confidence-weighted)
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

  // Benchmark counters (BENCHMARK_MODE)
  unsigned long benchDenoiseUs;
  unsigned long benchClassifyUs;
  unsigned long benchTotalUs;
  unsigned long benchWindows;

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
