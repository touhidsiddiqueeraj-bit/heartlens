#ifndef BATTERY_H
#define BATTERY_H

#include <Arduino.h>

class BatteryMonitor {
public:
  BatteryMonitor();
  bool begin(int pin);
  int readMillivolts();
  int readPercent();
  bool isLow();
  bool isCritical();

private:
  int _pin;
};

#endif
