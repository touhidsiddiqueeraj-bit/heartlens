#include "interpreter.h"
#include "Config.h"

// Rhythm class names (in order matching classifier output)
static const char* CLASS_NAMES[] = {
  "Normal Sinus Rhythm",
  "Atrial Fibrillation",
  "Premature Ventricular Contraction",
  "Tachycardia",
  "Bradycardia",
  "ST Abnormality"
};

// Urgency per class
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

  if (result.confidence < CONFIDENCE_HIGH) {
    out.urgency = Urgency::Error;
    out.message = "Signal unclear. Please reattach electrodes and try again.";
    return out;
  }

  int cls = result.classId;
  if (cls < 0 || cls > 5) cls = 0;

  // Debounce: don't re-alert within 30 seconds of a normal reading
  if (cls == 0) {
    _lastNormalTime = millis();
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
