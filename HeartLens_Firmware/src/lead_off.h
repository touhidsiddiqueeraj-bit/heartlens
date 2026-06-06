#ifndef LEAD_OFF_H
#define LEAD_OFF_H

#include <Arduino.h>

class LeadOffDetector {
public:
  LeadOffDetector();
  bool begin(int pinPos, int pinNeg);
  bool isDisconnected();
  void setThresholdMs(unsigned long ms);

private:
  int _pinPos;
  int _pinNeg;
  unsigned long _thresholdMs;
  unsigned long _reconnectMs;
  unsigned long _lastStableMs;
  bool _lastState;
  bool _confirmedDisconnected;
};

#endif
