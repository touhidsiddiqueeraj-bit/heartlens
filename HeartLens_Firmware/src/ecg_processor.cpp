#include "ecg_processor.h"
#include "Config.h"
#include "debug.h"
#include "models/denoiser_model.h"
#include "models/classifier_model.h"

#include <Arduino.h>
#include <cstring>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>

EcgProcessor::EcgProcessor()
    : _initialized(false), _arena(nullptr), _arenaSize(TENSOR_ARENA_SIZE),
      benchDenoiseUs(0), benchClassifyUs(0), benchTotalUs(0), benchWindows(0) {}

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

// ─── Signal quality gate ───────────────────────────────────────────
// Heuristic SQI: rejects flat/saturated buffers and buffers dominated by
// high-frequency (differential) energy — both indicate a corrupted or
// disconnected signal that a classifier would otherwise judge confidently.
static bool signalQualityOk(const int16_t* samples, int length) {
  if (length < 32) return false;

  int minV = samples[0], maxV = samples[0];
  long long sum = 0;
  for (int i = 0; i < length; i++) {
    if (samples[i] < minV) minV = samples[i];
    if (samples[i] > maxV) maxV = samples[i];
    sum += samples[i];
  }
  float mean = (float)sum / length;
  int range = maxV - minV;
  if (range < 8) return false;  // essentially flat

  // Saturation: >5% of samples pinned at extremes of the range
  int edge = range / 20 + 1;
  int saturated = 0;
  long double diffEnergy = 0.0, sigEnergy = 0.0;
  for (int i = 0; i < length; i++) {
    if (samples[i] <= minV + edge || samples[i] >= maxV - edge) saturated++;
    float s = (float)samples[i] - mean;
    sigEnergy += (double)s * s;
    if (i > 0) {
      float d = (float)(samples[i] - samples[i - 1]);
      diffEnergy += (double)d * d;
    }
  }
  if ((float)saturated / length > SQI_FLAT_FRACTION) return false;

  float noiseRatio = (sigEnergy > 0.0f) ? (float)(diffEnergy / sigEnergy) : 1.0f;
  if (noiseRatio > SQI_NOISE_THRESHOLD * SQI_NOISE_THRESHOLD) return false;
  return true;
}

WindowResult EcgProcessor::runSlidingInference(int16_t* samples, int length,
                                                int windowSize, int stride) {
  WindowResult wr;
  memset(wr.classScores, 0, sizeof(wr.classScores));
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
  if (numClasses > 8) numClasses = 8;

  // Pre-compute per-buffer normalization (same as training: [-1, 1])
  float center = 0.0f;
  for (int i = 0; i < length; i++) center += samples[i];
  center /= length;
  float maxDev = 0.0f;
  for (int i = 0; i < length; i++) {
    float dev = (samples[i] > center) ? (samples[i] - center) : (center - samples[i]);
    if (dev > maxDev) maxDev = dev;
  }
  if (maxDev < 1.0f) maxDev = 1.0f;  // prevent division by zero on flat signal

  for (int start = 0; start + windowSize <= length; start += stride) {
    unsigned long t0 = micros();

    int8_t* dIn = denoiserInput->data.int8;
    int copyLen = (windowSize < denoiserInputLen) ? windowSize : denoiserInputLen;
    for (int i = 0; i < copyLen; i++) {
      // Normalize to [-1, 1] matching training pipeline
      int idx = start + i;
      float normalized = (samples[idx] - center) / maxDev;
      normalized = (normalized < -1.0f) ? -1.0f : (normalized > 1.0f) ? 1.0f : normalized;
      // Convert to int8 via model's quantization params
      float iScale = denoiserInput->params.scale;
      int iZero = denoiserInput->params.zero_point;
      dIn[i] = (int8_t)((int)(normalized / iScale + iZero + 0.5f));
    }

    if (denoiserInterp.Invoke() != kTfLiteOk) {
      Serial.println("[ECG] Denoiser invoke failed");
      continue;
    }
    unsigned long tDen = micros();

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
    float probs[8];
    float sumP = 0.0f;
    for (int i = 0; i < numClasses; i++) {
      float prob = dequantizeInt8(scores[i], classScale, classZeroPoint);
      prob = (prob < 0.0f) ? 0.0f : prob;  // clamp negative
      probs[i] = prob;
      sumP += prob;
    }
    if (sumP <= 0.0f) continue;
    for (int i = 0; i < numClasses; i++) probs[i] /= sumP;

    // Confidence-weighted accumulation: sum softmax probs per class
    // (replaces argmax majority voting — review #13)
    for (int i = 0; i < numClasses; i++) wr.classScores[i] += probs[i];
    wr.totalWindows++;

    unsigned long t1 = micros();
    benchDenoiseUs += (tDen - t0);
    benchClassifyUs += (t1 - tDen);
    benchTotalUs += (t1 - t0);
    benchWindows++;

    DEBUG_PRINTF("[ECG] Window %d: top class=%d conf=%.3f (%lu us)\n",
                 wr.totalWindows, 0, sumP, t1 - t0);
  }

  return wr;
}

InferenceResult EcgProcessor::runInference(int16_t* samples, int length) {
  InferenceResult result = {0, 0.0f, {0}, false, true};
  if (!_initialized) return result;

  // Signal-quality gate before classification (review #6): a corrupted
  // buffer must not reach the classifier, which could score it confidently.
  if (!signalQualityOk(samples, length)) {
    result.signalOk = false;
    Serial.println("[ECG] Signal quality gate rejected buffer");
    return result;
  }

  int ws = (length < MODEL_INPUT_SAMPLES) ? length : MODEL_INPUT_SAMPLES;
  WindowResult wr = runSlidingInference(samples, length, ws, INFERENCE_STRIDE);

  if (wr.totalWindows == 0) return result;

  // Confidence-weighted decision across windows
  int bestClass = 0;
  float bestScore = -1.0f;
  float total = 0.0f;
  int numClasses = 0;
  for (int i = 0; i < 8; i++) {
    total += wr.classScores[i];
    if (wr.classScores[i] > bestScore) {
      bestScore = wr.classScores[i];
      bestClass = i;
    }
    if (wr.classScores[i] > 0.0f) numClasses = i + 1;
  }
  if (total <= 0.0f) return result;

  result.classId = bestClass;
  result.confidence = bestScore / total;  // mean calibrated probability
  for (int i = 0; i < 8; i++) result.probs[i] = wr.classScores[i] / total;
  result.valid = true;

  Serial.printf("[ECG] Sliding inference: %d windows, weighted=class %d (%.1f%%)\n",
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
