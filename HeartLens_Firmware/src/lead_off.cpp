#include "lead_off.h"
#include "Config.h"

LeadOffDetector::LeadOffDetector()
  : _pinPos(-1), _pinNeg(-1), _thresholdMs(LEAD_OFF_THRESHOLD_MS),
    _lastStableMs(0), _lastState(false) {}

bool LeadOffDetector::begin(int pinPos, int pinNeg) {
  _pinPos = pinPos;
  _pinNeg = pinNeg;
  pinMode(_pinPos, INPUT);
  pinMode(_pinNeg, INPUT);
  _lastStableMs = millis();
  _lastState = false;
  return true;
}

bool LeadOffDetector::isDisconnected() {
  // LOFF pins go HIGH when electrode is disconnected
  bool posOff = digitalRead(_pinPos) == HIGH;
  bool negOff = digitalRead(_pinNeg) == HIGH;
  bool currentlyOff = posOff || negOff;

  if (currentlyOff != _lastState) {
    _lastStableMs = millis();
    _lastState = currentlyOff;
    return false;  // debounce
  }

  if (currentlyOff && (millis() - _lastStableMs > _thresholdMs)) {
    return true;  // confirmed disconnected
  }

  return false;
}

void LeadOffDetector::setThresholdMs(unsigned long ms) {
  _thresholdMs = ms;
}
