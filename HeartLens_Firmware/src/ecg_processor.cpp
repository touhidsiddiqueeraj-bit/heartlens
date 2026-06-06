#include "ecg_processor.h"
#include "Config.h"
#include "models/denoiser_model.h"
#include "models/classifier_model.h"

#include <Arduino.h>
#include <cstring>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>

EcgProcessor::EcgProcessor() : _initialized(false), _arena(nullptr), _arenaSize(TENSOR_ARENA_SIZE) {}

bool EcgProcessor::begin() {
  _arena = malloc(_arenaSize);
  if (!_arena) {
    Serial.println("[ECG] Arena alloc failed!");
    return false;
  }
  memset(_arena, 0, _arenaSize);
  _initialized = true;
  Serial.println("[ECG] Processor ready");
  printModelInfo();
  return true;
}

float EcgProcessor::dequantizeInt8(int8_t value, float scale, int zeroPoint) {
  return ((float)value - (float)zeroPoint) * scale;
}

WindowResult EcgProcessor::runSlidingInference(int16_t* samples, int length,
                                                int windowSize, int stride) {
  WindowResult wr;
  memset(wr.classVotes, 0, sizeof(wr.classVotes));
  wr.totalWindows = 0;

  if (!_initialized || length < windowSize) return wr;

  // Shared arena halves: first half for denoiser, second half for classifier
  uint8_t* arena1 = (uint8_t*)_arena;
  uint8_t* arena2 = (uint8_t*)_arena + _arenaSize / 2;
  int halfArena = _arenaSize / 2;

  tflite::AllOpsResolver resolver;

  // Load models once, reuse interpreters
  const tflite::Model* denoiserModel = tflite::GetModel(denoiser_dummy_tflite);
  const tflite::Model* classifierModel = tflite::GetModel(classifier_dummy_tflite);

  tflite::MicroInterpreter denoiserInterp(denoiserModel, resolver, arena1, halfArena);
  if (denoiserInterp.AllocateTensors() != kTfLiteOk) {
    Serial.println("[ECG] Denoiser alloc failed");
    return wr;
  }

  tflite::MicroInterpreter classifierInterp(classifierModel, resolver, arena2, halfArena);
  if (classifierInterp.AllocateTensors() != kTfLiteOk) {
    Serial.println("[ECG] Classifier alloc failed");
    return wr;
  }

  TfLiteTensor* denoiserInput = denoiserInterp.input(0);
  TfLiteTensor* denoiserOutput = denoiserInterp.output(0);
  TfLiteTensor* classInput = classifierInterp.input(0);
  TfLiteTensor* classOutput = classifierInterp.output(0);

  int denoiserInputLen = denoiserInput->dims->data[1];
  int classInputLen = classInput->dims->data[1];
  int numClasses = classOutput->dims->data[classOutput->dims->size - 1];

  for (int start = 0; start + windowSize <= length; start += stride) {
    unsigned long t0 = micros();

    // Map ADC samples (millivolts) to int8 model input
    int8_t* dIn = denoiserInput->data.int8;
    int copyLen = (windowSize < denoiserInputLen) ? windowSize : denoiserInputLen;
    for (int i = 0; i < copyLen; i++) {
      // Scale millivolts (0-3300) to int8 range centered at 0
      int idx = start + i;
      dIn[i] = (int8_t)((samples[idx] - 1650) >> 4);
    }

    if (denoiserInterp.Invoke() != kTfLiteOk) {
      Serial.println("[ECG] Denoiser invoke failed");
      continue;
    }

    // Copy denoiser output to classifier input
    int classCopyLen = (denoiserOutput->bytes < (size_t)classInputLen)
                         ? denoiserOutput->bytes : (size_t)classInputLen;
    memcpy(classInput->data.int8, denoiserOutput->data.int8, classCopyLen);

    if (classifierInterp.Invoke() != kTfLiteOk) {
      Serial.println("[ECG] Classifier invoke failed");
      continue;
    }

    // Proper dequantization of int8 softmax output
    float classScale = classOutput->params.scale;
    int classZeroPoint = classOutput->params.zero_point;

    int8_t* scores = classOutput->data.int8;
    float maxScore = -1e10f;
    int maxIdx = 0;
    for (int i = 0; i < numClasses; i++) {
      float prob = dequantizeInt8(scores[i], classScale, classZeroPoint);
      prob = (prob < 0.0f) ? 0.0f : prob;  // clamp negative
      if (prob > maxScore) {
        maxScore = prob;
        maxIdx = i;
      }
    }

    wr.classVotes[maxIdx]++;
    wr.totalWindows++;

    unsigned long t1 = micros();
    DEBUG_PRINTF("[ECG] Window %d: class=%d confidence=%.3f (%lu us)\n",
                 wr.totalWindows, maxIdx, maxScore, t1 - t0);
  }

  return wr;
}

InferenceResult EcgProcessor::runInference(int16_t* samples, int length) {
  InferenceResult result = {0, 0.0f, false};
  if (!_initialized) return result;

  int ws = (length < MODEL_INPUT_SAMPLES) ? length : MODEL_INPUT_SAMPLES;
  WindowResult wr = runSlidingInference(samples, length, ws, INFERENCE_STRIDE);

  if (wr.totalWindows == 0) return result;

  // Majority vote across windows
  int bestClass = 0;
  int bestVotes = 0;
  int totalVotes = 0;
  for (int i = 0; i < 6; i++) {
    totalVotes += wr.classVotes[i];
    if (wr.classVotes[i] > bestVotes) {
      bestVotes = wr.classVotes[i];
      bestClass = i;
    }
  }

  result.classId = bestClass;
  result.confidence = (float)bestVotes / (float)wr.totalWindows;
  result.valid = true;

  Serial.printf("[ECG] Sliding inference: %d windows, majority=class %d (%.1f%%)\n",
                wr.totalWindows, bestClass, result.confidence * 100.0f);

  return result;
}

void EcgProcessor::printModelInfo() const {
  auto* dm = tflite::GetModel(denoiser_dummy_tflite);
  auto* cm = tflite::GetModel(classifier_dummy_tflite);
  Serial.printf("[ECG] Denoiser: %zu bytes, %d version\n",
                denoiser_dummy_tflite_len, dm->version());
  Serial.printf("[ECG] Classifier: %zu bytes, %d version\n",
                classifier_dummy_tflite_len, cm->version());
}
