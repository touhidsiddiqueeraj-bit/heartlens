#include "ecg_processor.h"
#include "Config.h"
#include "models/denoiser_model.h"
#include "models/classifier_model.h"

#include <Arduino.h>
#include <cstring>

// TFLite Micro headers — adjust include path if needed
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

InferenceResult EcgProcessor::runInference(int16_t* samples, int length) {
  InferenceResult result = {0, 0.0f, false};
  if (!_initialized) return result;

  Serial.println("[ECG] Running denoiser...");
  unsigned long t0 = micros();

  // ─── Stage 1: Load denoiser model ──────────────────────────────
  tflite::AllOpsResolver denoiserResolver;
  const tflite::Model* denoiserModel = tflite::GetModel(denoiser_dummy_tflite);
  if (denoiserModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ECG] Denoiser schema mismatch: %d\n", denoiserModel->version());
    return result;
  }

  tflite::MicroInterpreter denoiserInterp(
    denoiserModel, denoiserResolver, (uint8_t*)_arena, _arenaSize / 2
  );
  if (denoiserInterp.AllocateTensors() != kTfLiteOk) {
    Serial.println("[ECG] Denoiser alloc failed");
    return result;
  }

  TfLiteTensor* denoiserInput = denoiserInterp.input(0);
  TfLiteTensor* denoiserOutput = denoiserInterp.output(0);

  // Map ADC samples (0-4095) to int8 (-128 to 127)
  int8_t* dIn = denoiserInput->data.int8;
  for (int i = 0; i < length && i < denoiserInput->dims->data[1]; i++) {
    dIn[i] = (int8_t)((samples[i] - 2048) >> 4);  // scale to int8
  }

  if (denoiserInterp.Invoke() != kTfLiteOk) {
    Serial.println("[ECG] Denoiser invoke failed");
    return result;
  }

  unsigned long t1 = micros();
  Serial.printf("[ECG] Denoiser done: %lu us\n", t1 - t0);

  // ─── Stage 2: Load classifier model ────────────────────────────
  tflite::AllOpsResolver classifierResolver;
  const tflite::Model* classifierModel = tflite::GetModel(classifier_dummy_tflite);
  if (classifierModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ECG] Classifier schema mismatch: %d\n", classifierModel->version());
    return result;
  }

  // Use second half of arena
  uint8_t* arena2 = (uint8_t*)_arena + _arenaSize / 2;
  tflite::MicroInterpreter classifierInterp(
    classifierModel, classifierResolver, arena2, _arenaSize / 2
  );
  if (classifierInterp.AllocateTensors() != kTfLiteOk) {
    Serial.println("[ECG] Classifier alloc failed");
    return result;
  }

  TfLiteTensor* classInput = classifierInterp.input(0);
  TfLiteTensor* classOutput = classifierInterp.output(0);

  // Copy denoiser output to classifier input
  memcpy(classInput->data.int8, denoiserOutput->data.int8,
         min(denoiserOutput->bytes, classInput->bytes));

  if (classifierInterp.Invoke() != kTfLiteOk) {
    Serial.println("[ECG] Classifier invoke failed");
    return result;
  }

  unsigned long t2 = micros();
  Serial.printf("[ECG] Classifier done: %lu us\n", t2 - t1);
  Serial.printf("[ECG] Total inference: %lu us\n", t2 - t0);

  // ─── Extract result ────────────────────────────────────────────
  int8_t* scores = classOutput->data.int8;
  int numClasses = classOutput->dims->data[classOutput->dims->size - 1];
  float maxScore = -128.0f;
  int maxIdx = 0;
  for (int i = 0; i < numClasses; i++) {
    float normalized = (scores[i] + 128.0f) / 255.0f;
    if (normalized > maxScore) {
      maxScore = normalized;
      maxIdx = i;
    }
  }

  result.classId = maxIdx;
  result.confidence = maxScore;
  result.valid = true;
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
