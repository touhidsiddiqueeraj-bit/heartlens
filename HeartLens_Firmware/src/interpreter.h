#ifndef INTERPRETER_H
#define INTERPRETER_H

#include <Arduino.h>
#include "ecg_processor.h"

enum class Urgency {
  None,
  Medium,
  High,
  Error
};

struct OutputMessage {
  Urgency urgency;
  String message;
};

class Interpreter {
public:
  Interpreter();
  OutputMessage interpret(const InferenceResult& result);
  void setLastNormalTime(unsigned long ms);

private:
  unsigned long _lastNormalTime;
};

#endif
