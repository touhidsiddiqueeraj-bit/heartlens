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
  void* _oled;  // Adafruit_SSD1306 pointer (opaque to avoid include here)
  uint8_t _addr;
};

#endif
