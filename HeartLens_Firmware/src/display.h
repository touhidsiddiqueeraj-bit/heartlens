#ifndef DISPLAY_H
#define DISPLAY_H

#include <Arduino.h>
#include "interpreter.h"

class Display {
public:
  Display();
  bool begin(int sda, int scl, uint8_t addr);
  void showMessage(const String& text, Urgency urgency);
  void showBattery(int percent);
  void showSplash();
  void clear();
  void showSignalUnclear();
  void showIdle();

private:
  bool _initialized;
  void* _oled;
  uint8_t _addr;
  int _lastBatteryPercent;  // dirty-flag: avoid redundant redraws

  uint16_t urgencyColor(Urgency urgency);
  bool i2cPing();
};

#endif
