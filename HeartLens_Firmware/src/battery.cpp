#include "battery.h"
#include "Config.h"
#include <Arduino.h>

BatteryMonitor::BatteryMonitor() : _pin(-1) {}

bool BatteryMonitor::begin(int pin) {
  _pin = pin;
  pinMode(_pin, INPUT);
  analogReadResolution(12);
  return true;
}

int BatteryMonitor::readMillivolts() {
  int raw = analogRead(_pin);
  float vOut = (raw / 4095.0f) * ADC_VREF * 1000.0f;
  // Voltage divider: Vout = Vbat * R2 / (R1 + R2)
  float vBat = vOut * (BAT_DIVIDER_R1 + BAT_DIVIDER_R2) / BAT_DIVIDER_R2;
  return (int)vBat;
}

int BatteryMonitor::readPercent() {
  int mv = readMillivolts();
  // LiPo: 4.2V = 100%, 3.3V = 0%
  if (mv >= 4200) return 100;
  if (mv <= 3300) return 0;
  return ((mv - 3300) * 100) / (4200 - 3300);
}

bool BatteryMonitor::isLow() {
  return readMillivolts() < BAT_WARN_MV;
}

bool BatteryMonitor::isCritical() {
  return readMillivolts() < BAT_STOP_MV;
}
