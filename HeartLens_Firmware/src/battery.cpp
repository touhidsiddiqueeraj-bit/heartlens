#include "battery.h"
#include "Config.h"
#include <Arduino.h>
#include <esp_adc_cal.h>

// LiPo discharge lookup table (mV → percent)
// Typical 3.7V LiPo curve measured at 0.5C discharge
static const int LIPO_MV[] = {
  3300, 3400, 3500, 3550, 3600, 3650, 3700, 3720,
  3740, 3760, 3780, 3800, 3820, 3840, 3860, 3880,
  3900, 3920, 3940, 3960, 3980, 4000, 4050, 4100, 4200
};
static const int LIPO_PCT[] = {
  0, 2, 5, 8, 12, 16, 21, 26,
  31, 36, 42, 48, 54, 60, 65, 70,
  75, 79, 83, 87, 90, 93, 96, 98, 100
};
static const int LIPO_TABLE_SIZE = sizeof(LIPO_MV) / sizeof(LIPO_MV[0]);

BatteryMonitor::BatteryMonitor() : _pin(-1) {}

bool BatteryMonitor::begin(int pin) {
  _pin = pin;
  pinMode(_pin, INPUT);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  return true;
}

int BatteryMonitor::readMillivolts() {
  int raw = 0;
  for (int i = 0; i < BAT_ADC_SAMPLES; i++) {
    raw += analogRead(_pin);
    delayMicroseconds(100);
  }
  raw /= BAT_ADC_SAMPLES;

  float vOut = (raw / 4095.0f) * ADC_VREF * 1000.0f;
  float vBat = vOut * (BAT_DIVIDER_R1 + BAT_DIVIDER_R2) / BAT_DIVIDER_R2;
  return (int)vBat;
}

int BatteryMonitor::readPercent() {
  int mv = readMillivolts();

  if (mv <= LIPO_MV[0]) return 0;
  if (mv >= LIPO_MV[LIPO_TABLE_SIZE - 1]) return 100;

  // Interpolate from lookup table
  for (int i = 0; i < LIPO_TABLE_SIZE - 1; i++) {
    if (mv >= LIPO_MV[i] && mv < LIPO_MV[i + 1]) {
      float frac = (float)(mv - LIPO_MV[i]) / (float)(LIPO_MV[i + 1] - LIPO_MV[i]);
      return LIPO_PCT[i] + (int)(frac * (LIPO_PCT[i + 1] - LIPO_PCT[i]));
    }
  }

  return 0;
}

bool BatteryMonitor::isLow() {
  return readMillivolts() < BAT_WARN_MV;
}

bool BatteryMonitor::isCritical() {
  return readMillivolts() < BAT_STOP_MV;
}
