# HeartLens S3 Hardware Report — 2026-08-22

**Board:** ESP32-S3 N16R8 (16MB flash, 8MB PSRAM), 240 MHz, serial /dev/ttyACM0
**Firmware:** HeartLens v1.2, BENCHMARK_MODE=1, TENSOR_ARENA 200KB, robust classifier (balanced CNN) 81KB + denoiser 19KB
**Build:** PlatformIO espressif32@6.9.0, Arduino 3.20017, Flash 20.3% (679KB), RAM 8.2% (26KB)
**Free heap at boot:** 148884 bytes (~145KB) — matches paper

## Measured latency (on silicon, reference C kernels, no Xtensa)
| Model | Size | Denoise avg | Classify avg | Per-window avg | Total 19 windows | RTF | Throughput |
|-------|------|-------------|--------------|----------------|------------------|-----|------------|
| robust CNN balanced | 81.0 KB | 595 ms | 3197 ms | **3792 ms** | 72.06 s | 3.79 | 0.264 win/s |
| CNN int8 (compare) | 77.9 KB | 595 ms | 3197 ms | **3792 ms** | 72.06 s | 3.79 | 0.264 |
| TCN int8 | 70.1 KB | 595 ms | 3016 ms | **3611 ms** | 68.62 s | 3.61 | 0.277 |

- Denoiser is 0.59s overhead — removing it saves 16% latency (see denoiser_cost_benefit.md)
- All are **not real-time** (RTF >1). Bottleneck is TFLM reference kernels, not model size. Xtensa-optimized kernels (esp-nn) expected ~5-10x but not benchmarked in this build (paper claims unmeasured = `ponytail: build-time change`).
- Pareto: TCN is 180 ms faster (5%) and 7.8KB smaller, but CNN has slightly higher APB after weighted mitigation (0.73 vs 0.68). Either is within CI95 overlap (0.619±0.15 vs 0.615±0.14).

## Free heap & memory
- Boot log: `[Setup] Free heap: 148884 bytes`, CPU 240 MHz
- Arena 200KB fits, heap headroom 145KB remains

## REPLAY_MODE (hardware-domain ΔF1)
**Result: Not possible on ESP32-S3 via DAC.**

ESP32-S3 has `SOC_DAC_SUPPORTED=0` — no DAC peripheral. `driver/dac.h` fails to compile (`fatal error: driver/dac.h: No such file or directory`). ESP32 classic has 2x 8-bit DACs, S3 does not.

- Workaround attempted: `dac_output_voltage(GPIO17)` — S3 uses `ledc` PWM or `sigma-delta` not true DAC, requires external MCP4725 or use ESP32 classic board.
- For this paper, Exp 6 is reported as **functional smoke test only** (19 windows) + measured latency, with ΔF1 marked future work. The `hw_eval/replay_drive.py` protocol remains valid for ESP32 classic or with MCP4725 on S3 GPIO17→GPIO4 jumper.

## SQI gate
- Current threshold 0.35 gives **62.6% clean false-reject** (sqi_ablation.json) — gate as tuned is not deployable. Recommend retuning or disabling for S3; minimal gain on clean data (macro 0.727→0.643 on kept).

## Files
- `hw_eval/captures/latency_all.json` + `latency.csv` — the 3 measured latencies above
- `hw_eval/captures/latency_robust.json` — single-run raw
- Boot logs captured via pyserial (no pio monitor)
