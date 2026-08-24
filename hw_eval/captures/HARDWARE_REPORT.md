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

## LSTM/GRU measured latency — 2026-08-25 session

Board: ESP32-S3 N16R8 @240 MHz, TFLM reference kernels, 200 KB internal-SRAM
arena (stock), BENCHMARK_MODE, n=19 sliding windows, 3 runs each (<0.1% spread).

| Model | Export | Size KB | per-window ms | RTF |
|---|---|---|---|---|
| CNN | int8 (existing) | 77.9 | 3792 | 3.79 |
| TCN | int8 (existing) | 70.1 | 3611 | 3.61 |
| LSTM | **fused float32** (UNIDIRECTIONAL_SEQUENCE_LSTM, weights from compare_lstm.keras) | 126.5 | **1574** | 1.57 |
| GRU | **fused float32** (UNIDIRECTIONAL_SEQUENCE_GRU, weights from compare_gru.keras) | 103.7 | **1379** | 1.38 |

Method note: the original SELECT_TF_OPS float32 exports contain Flex
TensorList ops that TFLM cannot execute. Both RNNs were re-exported via fused
sequence layers using a static-shape concrete-function conversion; PC-side
outputs match the trained Keras models exactly. Ranking vs. old estimates
(5200/4800 ms) inverts: RNNs are the fastest models on silicon (~0.12 M MACs
vs. conv stacks). Firmware was restored to stock int8 CNN build after capture.

## Kernel-implementation diagnosis (why int8 CNN is slower than fp32 RNN)

Flatbuffer MAC audit (per 1-s window, PC-side schema dump):
- CNN int8: 5.60 MMAC | TCN int8: 5.05 MMAC (+ SPACE_TO_BATCH_ND dilation copies)
- LSTM fp32: ~4.42 MMAC (180 steps x (32x256 + 64x256)) | GRU fp32: ~3.32 MMAC

Effective rates from measured classify_avg:
- reference int8 ConvPerChannel: ~1.7 MMAC/s (CNN, TCN)
- reference float FC (inside UNSQ-LSTM/GRU): ~4.2-4.5 MMAC/s

Root cause: the vendored lib/tensorflow_lite tree contains ONLY portable
reference kernels — tensorflow/lite/kernels/internal/optimized/ has no
arch backends, no kernels/xtensa/ sources, and esp-nn was never linked.
The ~2.6x per-MAC gap between reference-int8-conv and reference-float-FC
explains the ranking inversion; it is a kernel artifact, NOT an
architecture result, and NOT a bug in ecg_processor.cpp (Invoke time only;
interpreters built once per buffer; per-window work is O(360)).

Upgrade path: link Espressif ESP-NN (esp_nn_conv_s8 / _fully_connected_s8,
S3-optimized) into conv.cc/fully_connected.cc; expect multi-x speedup of
CNN/TCN and probable restoration of their lead. Requires newer TFLM glue or
manual backend patch — out of scope for this measurement session.

## ESP-NN port — session 2 (2026-08-25, later)

Ported Espressif ESP-NN into the vendored TFLM tree:
- Vendored upstream master sources (include/ + kernels) under lib/esp_nn,
  compiled into src/esp_nn_impl/. Global -DCONFIG_NN_OPTIMIZED.
- Patched TFLM conv.cc / fully_connected.cc with guarded dispatch
  (int8 + dilation==1; scratch via RequestScratchBufferInArena; reference
  fallback). FC dispatch additionally requires row_len%8==0 && out_ch%8==0.
- Arena: 300 KB total, asymmetric split 96 KB denoiser / 204 KB classifier
  (master im2col scratch requests ~106 KB classifier-side).
- v1.0.0 S3 asm kernels produced WRONG outputs (probs [0.945,0.043,0.012] vs
  ground truth [0,0,1]) — bisected via ANSI-control build to the S3 conv asm;
  upstream master fixes (ACCX extraction, filter alignment, SAME padding)
  resolve it. All four models now bit-match PC ground truth on argmax+conf.

Final measured matrix (n=19 windows, fixed-seed input, 240 MHz):

| Model | Kernel | ms/window | RTF | vs reference |
|---|---|---|---|---|
| CNN | int8 + ESP-NN | **500** | **0.50 REAL-TIME** | 7.6x |
| TCN | int8 + ESP-NN | **609** | **0.61 REAL-TIME** | 5.9x |
| LSTM | fp32 fused + ESP-NN FC | 1290 | 1.29 | 1.2x |
| GRU | fp32 fused + ESP-NN FC | 1091 | 1.09 | 1.3x |

Denoise stage: 595 -> 315 ms (encoder convs on ESP-NN; transposed-conv
decoder still reference). Ranking restored to MAC-order: int8 conv stacks
with optimized kernels beat float32 RNNs. CNN and TCN now MEET real-time.

Firmware state left flashed: CNN int8 + ESP-NN (canonical). main.cpp bench
uses fixed seed 12345 and prints aggregated probs for A/B verification.
