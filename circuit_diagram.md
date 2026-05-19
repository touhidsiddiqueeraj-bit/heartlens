# HeartLens AI — System Circuit Diagram

```mermaid
block-beta
  columns 4

  block:power:3
    columns 3
    USB_C["USB-C 5V"]
    TP4056["TP4056<br/>LiPo Charger"]
    LIPO["LiPo 3.7V<br/>500-1000 mAh"]

    AMS1117["AMS1117-3.3<br/>LDO Regulator"]
    V3_RAIL["3.3V Rail"]
  end

  block:analog:3
    columns 3
    ELECTRODES["Electrodes<br/>LA / RA / RL"]
    AD8232["AD8232<br/>ECG AFE"]
    ESP_ADC["ESP32<br/>GPIO34 (ADC1_CH6)"]
  end

  block:digital:3
    columns 3
    DENOISER["LSTM Denoiser<br/>TFLite Micro"]
    CNN["1D-CNN Classifier<br/>TFLite Micro"]
    INTERP["Rule Interpreter<br/>C++"]
  end

  block:output:3
    columns 3
    SSD1306["SSD1306 OLED<br/>128x64 I2C"]
    UART["UART Debug<br/>GPIO1(TX)/GPIO3(RX)"]
    LED["Status LED<br/>GPIO2"]
  end

  block:monitor:2
    columns 2
    BAT_DIV["Battery Divider<br/>10k + 10k → GPIO35"]
    LEAD_OFF["Lead-Off Detect<br/>LOFF+ → GPIO32<br/>LOFF- → GPIO33"]
  end

  USB_C --> TP4056
  TP4056 --> LIPO
  LIPO --> AMS1117
  AMS1117 --> V3_RAIL

  V3_RAIL --> AD8232
  V3_RAIL --> ESP_ADC
  V3_RAIL --> SSD1306

  ELECTRODES --> AD8232
  AD8232 --> ESP_ADC
  AD8232 --> LEAD_OFF

  ESP_ADC --> DENOISER
  DENOISER --> CNN
  CNN --> INTERP
  INTERP --> SSD1306
  INTERP --> LED

  LIPO --> BAT_DIV
  BAT_DIV --> ESP_ADC
```

## Pin Connection Table

| Component | Pin | Connects To | Notes |
|-----------|-----|-------------|-------|
| Electrode LA | — | AD8232 IN+ | Positive differential input |
| Electrode RA | — | AD8232 IN- | Negative differential input |
| Electrode RL | — | AD8232 RLD | Right-leg drive output |
| **AD8232** | OUTPUT | ESP32 GPIO34 | Filtered ECG signal (0.5-2.5V) |
| | LOFF+ | ESP32 GPIO32 | Lead-off detect positive |
| | LOFF- | ESP32 GPIO33 | Lead-off detect negative |
| | VCC | 3.3V rail | 100nF decoupling to GND |
| **SSD1306** | SDA | ESP32 GPIO21 | I2C data, 4.7k pull-up |
| | SCL | ESP32 GPIO22 | I2C clock, 4.7k pull-up |
| | VCC | 3.3V rail | 100nF decoupling |
| **TP4056** | VIN | USB-C VBUS | 5V charging input |
| | VOUT+ | LiPo BAT+ | 4.2V max |
| **AMS1117** | VIN | LiPo BAT+ | Input voltage |
| | VOUT | 3.3V rail | Regulated 3.3V |
| **Battery Divider** | — | ESP32 GPIO35 | 10k/10k divider |
| **ESP32** | EN | 10k pull-up to 3.3V | Reset button |

## Power Budget

| Component | Current (mA) |
|-----------|-------------|
| ESP32 (active, dual-core) | ~60 |
| AD8232 | ~0.5 |
| SSD1306 (average) | ~10 |
| AMS1117 quiescent | ~5 |
| **Total** | **~75 mA** |
| **Battery life (500 mAh)** | **~6.5 hours** |
