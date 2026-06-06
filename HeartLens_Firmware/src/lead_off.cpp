#include "lead_off.h"
#include "Config.h"

LeadOffDetector::LeadOffDetector()
  : _pinPos(-1), _pinNeg(-1), _thresholdMs(LEAD_OFF_THRESHOLD_MS),
    _reconnectMs(LEAD_OFF_RECONNECT_MS),
    _lastStableMs(0), _lastState(false), _confirmedDisconnected(false) {}

bool LeadOffDetector::begin(int pinPos, int pinNeg) {
  _pinPos = pinPos;
  _pinNeg = pinNeg;
  pinMode(_pinPos, INPUT);
  pinMode(_pinNeg, INPUT);
  _lastStableMs = millis();
  _lastState = false;
  _confirmedDisconnected = false;
  return true;
}

bool LeadOffDetector::isDisconnected() {
  bool posOff = digitalRead(_pinPos) == HIGH;
  bool negOff = digitalRead(_pinNeg) == HIGH;
  bool currentlyOff = posOff || negOff;
  unsigned long now = millis();

  if (currentlyOff != _lastState) {
    _lastStableMs = now;
    _lastState = currentlyOff;
    if (!currentlyOff) {
      // Transition from off → on: hysteresis hold
      return _confirmedDisconnected;
    }
    return false;  // debounce on disconnect
  }

  if (currentlyOff) {
    if (now - _lastStableMs > _thresholdMs) {
      _confirmedDisconnected = true;
      return true;
    }
  } else {
    // Stable connected — clear confirmed state
    _confirmedDisconnected = false;
  }

  return _confirmedDisconnected;
}

void LeadOffDetector::setThresholdMs(unsigned long ms) {
  _thresholdMs = ms;
}
