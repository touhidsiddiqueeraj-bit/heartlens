#include "display.h"
#include "Config.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

// Colors for each urgency level
static const uint16_t COLOR_NORMAL = SSD1306_WHITE;
static const uint16_t COLOR_MEDIUM = SSD1306_WHITE;
static const uint16_t COLOR_HIGH = SSD1306_WHITE;
static const uint16_t COLOR_ERROR = SSD1306_WHITE;

Display::Display() : _initialized(false), _oled(nullptr), _addr(OLED_I2C_ADDR) {}

bool Display::begin(int sda, int scl, uint8_t addr) {
  _addr = addr;
  Wire.begin(sda, scl);
  _oled = new Adafruit_SSD1306(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
  if (!_oled) return false;

  auto* oled = static_cast<Adafruit_SSD1306*>(_oled);
  if (!oled->begin(SSD1306_SWITCHCAPVCC, addr)) {
    Serial.println("[Display] SSD1306 init failed!");
    return false;
  }

  oled->clearDisplay();
  oled->setTextWrap(true);
  _initialized = true;
  return true;
}

void Display::showSplash() {
  if (!_initialized) return;
  auto* oled = static_cast<Adafruit_SSD1306*>(_oled);
  oled->clearDisplay();

  oled->setTextSize(1);
  oled->setTextColor(SSD1306_WHITE);
  oled->setCursor(8, 8);
  oled->println("HeartLens AI");
  oled->setCursor(8, 20);
  oled->println("Edge ECG Monitor");
  oled->setCursor(8, 36);
  oled->println("Starting...");

  // Draw a simple heartbeat line
  for (int x = 0; x < 128; x++) {
    int y = 54 + (int)(4 * sin(x * 0.2 + millis() / 1000.0));
    oled->drawPixel(x, y, SSD1306_WHITE);
  }

  oled->display();
}

void Display::showMessage(const String& text, Urgency urgency) {
  if (!_initialized) return;
  auto* oled = static_cast<Adafruit_SSD1306*>(_oled);
  oled->clearDisplay();

  // Urgency indicator bar at top
  uint16_t barColor;
  switch (urgency) {
    case Urgency::None:     barColor = SSD1306_WHITE; break;
    case Urgency::Medium:   barColor = SSD1306_WHITE; break;
    case Urgency::High:     barColor = SSD1306_WHITE; break;
    default:                barColor = SSD1306_WHITE; break;
  }

  // Top status bar (inverted)
  oled->fillRect(0, 0, 128, 10, barColor);
  oled->setTextColor(SSD1306_BLACK);
  oled->setCursor(2, 1);
  oled->setTextSize(1);

  switch (urgency) {
    case Urgency::None:   oled->print("NORMAL"); break;
    case Urgency::Medium: oled->print("CAUTION"); break;
    case Urgency::High:   oled->print("ALERT"); break;
    default:              oled->print("ERROR"); break;
  }

  // Main message — word-wrap manually
  oled->setTextColor(SSD1306_WHITE);
  oled->setCursor(2, 16);
  oled->setTextSize(1);

  int maxChars = 21;
  if (text.length() <= maxChars) {
    oled->println(text);
  } else {
    // Simple word wrap
    int lastSpace = -1;
    for (int i = 0; i < text.length(); i++) {
      if (text[i] == ' ') lastSpace = i;
    }
    if (lastSpace > 0 && lastSpace <= maxChars) {
      oled->println(text.substring(0, lastSpace));
      oled->println(text.substring(lastSpace + 1));
    } else {
      oled->println(text.substring(0, maxChars));
      oled->println(text.substring(maxChars));
    }
  }

  oled->display();
}

void Display::showBattery(int percent) {
  if (!_initialized) return;
  auto* oled = static_cast<Adafruit_SSD1306*>(_oled);

  // Draw battery icon in top-right corner
  int bx = 104, by = 1;
  oled->drawRect(bx, by, 18, 8, SSD1306_WHITE);     // battery body
  oled->drawRect(bx + 18, by + 2, 2, 4, SSD1306_WHITE);  // terminal

  int fillWidth = (percent * 14) / 100;
  if (fillWidth > 0) {
    oled->fillRect(bx + 2, by + 2, fillWidth, 4, SSD1306_WHITE);
  }

  oled->display();
}

void Display::clear() {
  if (!_initialized) return;
  auto* oled = static_cast<Adafruit_SSD1306*>(_oled);
  oled->clearDisplay();
  oled->display();
}

void Display::showSignalUnclear() {
  showMessage("Signal unclear", Urgency::Error);
}

void Display::showIdle() {
  if (!_initialized) return;
  auto* oled = static_cast<Adafruit_SSD1306*>(_oled);
  oled->clearDisplay();
  oled->setCursor(8, 24);
  oled->setTextSize(1);
  oled->setTextColor(SSD1306_WHITE);
  oled->println("Place electrodes");
  oled->setCursor(8, 36);
  oled->println("to begin...");
  oled->display();
}
