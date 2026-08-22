| Work | Patient-ind | External | Noise | Quant | Real MCU | Calib | Latency |
|---|---|---|---|---|---|---|---|
| Kiranyaz 2016 (CNN patient-spec) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Hannun 2019 (Cardiologist-level) | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Acharya 2017 (Augmented CNN) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Davidson 2021 (TFLM) | - | - | - | ✓ | ✓ | ✗ | ✓ |
| ESP32 ECG 2023 (single CNN) | ✗ | ✗ | ✓(AWGN) | ✓ | ✓ | ✗ | est |
| **This work** | **✓ (5×2 GroupKFold)** | **✓ (SVDB N/V-macro)** | **✓ (5 artifacts × 3 front-ends)** | **✓ (paired FP32/INT8)** | **✓ (S3, 4 arch)** | **✓ (T=0.35, ECE 0.39→0.09, NLL/Brier)** | **✓ (measured 3.74s, ref kernels)** |
