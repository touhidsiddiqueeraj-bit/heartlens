#include "interpreter.h"
#include "Config.h"

static const char* CLASS_NAMES[] = {
  "Normal Sinus Rhythm",
  "Atrial Premature Beat (APB)",
  "Premature Ventricular Contraction"
};

static const Urgency CLASS_URGENCY[] = {
  Urgency::None,    // Normal
  Urgency::Medium,  // APB
  Urgency::Medium   // PVC
};

static const char* OUTPUT_MESSAGES[] = {
  "Heart rhythm looks normal.",
  "Unusual rhythm detected. Please see a doctor soon.",
  "Unusual rhythm detected. Please see a doctor soon."
};

// Temperature-scaled softmax on probabilities (review #12).
// q_i = p_i^(1/T) / sum_j p_j^(1/T); T=1.0 is identity.
static void applyTemperature(float* probs, int n, float temperature) {
  if (temperature <= 0.0f) temperature = 1.0f;
  float inv = 1.0f / temperature;
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    if (probs[i] > 0.0f) probs[i] = powf(probs[i], inv);
    sum += probs[i];
  }
  if (sum > 0.0f) {
    for (int i = 0; i < n; i++) probs[i] /= sum;
  }
}

Interpreter::Interpreter() : _lastNormalTime(0) {}

OutputMessage Interpreter::interpret(const InferenceResult& result) {
  OutputMessage out;

  if (!result.valid) {
    out.urgency = Urgency::Error;
    out.message = "Signal error. Please reattach electrodes and try again.";
    return out;
  }

  if (!result.signalOk) {
    out.urgency = Urgency::Error;
    out.message = "Signal unclear. Please reattach electrodes and try again.";
    return out;
  }

  if (result.confidence < CONFIDENCE_LOW) {
    out.urgency = Urgency::Error;
    out.message = "Signal unclear. Please reattach electrodes and try again.";
    return out;
  }

  int cls = result.classId;
  if (cls < 0 || cls >= NUM_CLASSES) cls = 0;

  // Calibrate the full class distribution via temperature scaling (review #12)
  float probs[NUM_CLASSES];
  float sum = 0.0f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    probs[i] = (result.probs[i] > 0.0f) ? result.probs[i] : 0.0f;
    sum += probs[i];
  }
  if (sum <= 0.0f) probs[cls] = 1.0f;  // degenerate: fall back to argmax
  applyTemperature(probs, NUM_CLASSES, CALIB_TEMPERATURE);
  float confCal = probs[cls];

  // Low confidence (between LOW and HIGH) — still show result but indicate uncertainty
  if (confCal < CONFIDENCE_HIGH) {
    out.urgency = CLASS_URGENCY[cls];
    out.message = String("Possible: ") + OUTPUT_MESSAGES[cls];
    Serial.printf("[Interp] Low conf: class=%s (%d)  confidence=%.3f  urgency=%d\n",
                  CLASS_NAMES[cls], cls, confCal, (int)out.urgency);
    return out;
  }

  // Debounce: don't re-alert within NORMAL_DEBOUNCE_MS of a normal reading
  unsigned long now = millis();
  if (cls == 0) {
    _lastNormalTime = now;
  } else {
    if (_lastNormalTime > 0 && (now - _lastNormalTime < NORMAL_DEBOUNCE_MS)) {
      // Within debounce window after a normal reading — downgrade urgency
      out.urgency = Urgency::None;
      out.message = "Heart rhythm looks normal.";
      Serial.printf("[Interp] Debounce active: %lu ms since normal\n",
                     now - _lastNormalTime);
      return out;
    }
  }

  out.urgency = CLASS_URGENCY[cls];
  out.message = OUTPUT_MESSAGES[cls];

  Serial.printf("[Interp] Class=%s (%d)  Confidence=%.3f  Urgency=%d\n",
                CLASS_NAMES[cls], cls, confCal, (int)out.urgency);

  return out;
}

void Interpreter::setLastNormalTime(unsigned long ms) {
  _lastNormalTime = ms;
}
