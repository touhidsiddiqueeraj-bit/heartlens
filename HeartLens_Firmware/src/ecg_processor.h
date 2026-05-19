#ifndef ECG_PROCESSOR_H
#define ECG_PROCESSOR_H

#include <cstdint>

struct InferenceResult {
  int classId;           // 0-5
  float confidence;      // 0.0-1.0
  bool valid;            // false if inference failed
};

class EcgProcessor {
public:
  EcgProcessor();
  bool begin();
  InferenceResult runInference(int16_t* samples, int length);
  void printModelInfo() const;

private:
  bool _initialized;
  void* _arena;
  int _arenaSize;

  bool loadModel(const unsigned char modelData[], int modelSize,
                 const char* name, void* arena, int arenaSize,
                 int& inputSize, int& outputSize);
};

#endif
