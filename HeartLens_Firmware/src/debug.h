#ifndef DEBUG_H
#define DEBUG_H

#include <Arduino.h>

#ifdef DEBUG_ENABLED
  #define DEBUG_PRINT(...)     Serial.print(__VA_ARGS__)
  #define DEBUG_PRINTLN(...)   Serial.println(__VA_ARGS__)
  #define DEBUG_PRINTF(...)    Serial.printf(__VA_ARGS__)
  #define DEBUG_TIMESTAMP(msg) do { \
    Serial.printf("[%lu] %s\n", millis(), msg); \
  } while(0)
#else
  #define DEBUG_PRINT(...)
  #define DEBUG_PRINTLN(...)
  #define DEBUG_PRINTF(...)
  #define DEBUG_TIMESTAMP(msg)
#endif

#endif // DEBUG_H
