#!/bin/bash
# setup_tflite_micro.sh
# Downloads TFLite Micro source for ESP32 into lib/tensorflow_lite
# Usage: ./setup_tflite_micro.sh
#
# Pinned to v0.8.0: newer TFLM removed AllOpsResolver / the micro/
# include layout this firmware targets. Upgrade requires porting
# ecg_processor.cpp to MicroMutableOpResolver.

set -e
FW_DIR="$(dirname "$0")"
LIB_DIR="$FW_DIR/lib/tensorflow_lite"
TMP_DIR=$(mktemp -d)
# Last commit containing all_ops_resolver.h (parent of the deprecation
# commit e344f4b6). Newer TFLM removed AllOpsResolver — upgrade requires
# porting ecg_processor.cpp to MicroMutableOpResolver.
PIN="${TFLM_PIN:-d5f70ceb5202f72efc856901c3418fff92b76f9e}"

echo "Downloading TFLite Micro ($PIN)..."
git clone --filter=blob:none --no-checkout \
  https://github.com/tensorflow/tflite-micro.git "$TMP_DIR/tflite-micro"
git -C "$TMP_DIR/tflite-micro" checkout "$PIN"

mkdir -p "$LIB_DIR"

# Copy only the needed subset for ESP32 inference
cp -r "$TMP_DIR/tflite-micro/tensorflow" "$LIB_DIR/tensorflow"
cp -r "$TMP_DIR/tflite-micro/third_party" "$LIB_DIR/third_party"
# third_party is Bazel-only metadata except the vendored headers we
# re-fetch below; drop the stubs so PIO does not compile them
for d in BUILD hexagon kissfft python_requirements.in \
         python_requirements.txt repo.bzl six.BUILD xtensa BUILD.bazel \
         BUILD.external BUILD.system build_defs.bzl workspace.bzl; do
  rm -rf "$LIB_DIR/third_party/$d"
done
# experimental/ contains audio/microfrontend code not used by inference
rm -rf "$LIB_DIR/tensorflow/lite/experimental"
# tests are not needed and can collide with sketch mains
find "$LIB_DIR/tensorflow" -name "*_test.cc" -delete
# non-ESP32 platform dirs (the TFLM make build compiles only the
# target's platform code; a raw PIO lib compiles everything)
for d in arc_custom arc_emsdp bluepill ceva chre cortex_m_corstone_300 \
         cortex_m_generic hexagon riscv32_mcu stm32f4 sparc; do
  rm -rf "$LIB_DIR/tensorflow/lite/micro/$d"
  rm -rf "$LIB_DIR/tensorflow/lite/micro/kernels/$d"
done
# foreign kernel backends (xtensa dir also dropped: its INT8REF variants
# re-register kernels unless built with XTENSA NN flags; reference
# kernels in kernels/ are used instead)
for d in arc_mli cmsis_nn ethos_u vexriscv test_data_generation xtensa; do
  rm -rf "$LIB_DIR/tensorflow/lite/micro/kernels/$d"
done
# benchmarks/examples/models pull in generated model data; unused
rm -rf "$LIB_DIR/tensorflow/lite/micro/benchmarks"
rm -rf "$LIB_DIR/tensorflow/lite/micro/examples"
rm -rf "$LIB_DIR/tensorflow/lite/micro/models"
rm -rf "$LIB_DIR/tensorflow/lite/micro/integration_tests"
rm -rf "$LIB_DIR/tensorflow/lite/micro/python"

# third_party/flatbuffers is a Bazel vendored dependency (not a git
# submodule in this revision) — fetch the pinned tarball
FB_COMMIT="a66de58af9565586832c276fbb4251fc416bf07f"
echo "Downloading flatbuffers ($FB_COMMIT)..."
curl -sL "https://github.com/google/flatbuffers/archive/${FB_COMMIT}.tar.gz" \
  -o "$TMP_DIR/flatbuffers.tar.gz"
tar -xzf "$TMP_DIR/flatbuffers.tar.gz" -C "$TMP_DIR"
rm -rf "$LIB_DIR/third_party/flatbuffers"
mkdir -p "$LIB_DIR/third_party/flatbuffers"
# only the headers are needed (the tarball also ships android/samples etc.)
cp -r "$TMP_DIR/flatbuffers-${FB_COMMIT}/include" \
      "$LIB_DIR/third_party/flatbuffers/"

# gemmlowp fixedpoint headers (also a Bazel-vendored download)
GL_COMMIT="719139ce755a0f31cbf1c37f7f98adcc7fc9f425"
echo "Downloading gemmlowp ($GL_COMMIT)..."
curl -sL "https://github.com/google/gemmlowp/archive/${GL_COMMIT}.zip" \
  -o "$TMP_DIR/gemmlowp.zip"
unzip -q "$TMP_DIR/gemmlowp.zip" -d "$TMP_DIR"
mkdir -p "$LIB_DIR/third_party/gemmlowp"
cp -r "$TMP_DIR/gemmlowp-${GL_COMMIT}/fixedpoint" \
      "$LIB_DIR/third_party/gemmlowp/"
mkdir -p "$LIB_DIR/third_party/gemmlowp/internal"
cp "$TMP_DIR/gemmlowp-${GL_COMMIT}/internal/detect_platform.h" \
   "$LIB_DIR/third_party/gemmlowp/internal/"

# kissfft (used by the microfrontend code that PIO compiles)
KF_COMMIT="33d9ad3bad3fe8f1fb43a4634f61ea9a40240534"
echo "Downloading kissfft ($KF_COMMIT)..."
curl -sL "https://github.com/mborgerding/kissfft/archive/${KF_COMMIT}.tar.gz" \
  -o "$TMP_DIR/kissfft.tar.gz"
tar -xzf "$TMP_DIR/kissfft.tar.gz" -C "$TMP_DIR"
rm -rf "$LIB_DIR/third_party/kissfft"
mv "$TMP_DIR/kissfft-${KF_COMMIT}" "$LIB_DIR/third_party/kissfft"
# only kiss_fft.{h,c} (+ COPYING) are used; prune tests/tools
rm -rf "$LIB_DIR/third_party/kissfft/test" "$LIB_DIR/third_party/kissfft/tools"
find "$LIB_DIR/third_party/kissfft" -name "*.c" ! -name "kiss_fft.c" -delete

# ruy (profiler/instrumentation.h is included by reference kernels)
RUY_COMMIT="d37128311b445e758136b8602d1bbd2a755e115d"
echo "Downloading ruy ($RUY_COMMIT)..."
curl -sL "https://github.com/google/ruy/archive/${RUY_COMMIT}.zip" \
  -o "$TMP_DIR/ruy.zip"
unzip -q "$TMP_DIR/ruy.zip" -d "$TMP_DIR"
rm -rf "$LIB_DIR/third_party/ruy"
mv "$TMP_DIR/ruy-${RUY_COMMIT}" "$LIB_DIR/third_party/ruy"
# TFLM only includes ruy/profiler/instrumentation.h — prune the rest
find "$LIB_DIR/third_party/ruy" -name "*_test.cc" -delete
find "$LIB_DIR/third_party/ruy" -name "test.cc" -delete
find "$LIB_DIR/third_party/ruy" -name "*.cc" ! -path "*/profiler/*" -delete
find "$LIB_DIR/third_party/ruy" -name "*.c" -delete

rm -rf "$TMP_DIR"
echo "TFLite Micro copied to $LIB_DIR"
echo "Size: $(du -sh "$LIB_DIR" | cut -f1)"
echo "Done."
