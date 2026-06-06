#include "interpreter.h"
#include "Config.h"

static const char* CLASS_NAMES[] = {
  "Normal Sinus Rhythm",
  "Atrial Fibrillation",
  "Premature Ventricular Contraction",
  "Tachycardia",
  "Bradycardia",
  "ST Abnormality"
};

static const Urgency CLASS_URGENCY[] = {
  Urgency::None,    // Normal
  Urgency::High,    // AFib
  Urgency::Medium,  // PVC
  Urgency::Medium,  // Tachy
  Urgency::Medium,  // Brady
  Urgency::High     // ST Abn
};

static const char* OUTPUT_MESSAGES[] = {
  "Heart rhythm looks normal.",
  "Irregular rhythm detected. Please seek medical attention.",
  "Unusual rhythm detected. Please see a doctor soon.",
  "Unusual rhythm detected. Please see a doctor soon.",
  "Unusual rhythm detected. Please see a doctor soon.",
  "Irregular rhythm detected. Please seek medical attention."
};

Interpreter::Interpreter() : _lastNormalTime(0) {}

OutputMessage Interpreter::interpret(const InferenceResult& result) {
  OutputMessage out;

  if (!result.valid) {
    out.urgency = Urgency::Error;
    out.message = "Signal error. Please reattach electrodes and try again.";
    return out;
  }

  if (result.confidence < CONFIDENCE_LOW) {
    out.urgency = Urgency::Error;
    out.message = "Signal unclear. Please reattach electrodes and try again.";
    return out;
  }

  int cls = result.classId;
  if (cls < 0 || cls > 5) cls = 0;

  // Low confidence (between LOW and HIGH) — still show result but indicate uncertainty
  if (result.confidence < CONFIDENCE_HIGH) {
    out.urgency = CLASS_URGENCY[cls];
    out.message = String("Possible: ") + OUTPUT_MESSAGES[cls];
    Serial.printf("[Interp] Low conf: class=%s (%d)  confidence=%.3f  urgency=%d\n",
                  CLASS_NAMES[cls], cls, result.confidence, (int)out.urgency);
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
                CLASS_NAMES[cls], cls, result.confidence, (int)out.urgency);

  return out;
}

void Interpreter::setLastNormalTime(unsigned long ms) {
  _lastNormalTime = ms;
}
