#!/bin/bash
# setup_tflite_micro.sh
# Downloads TFLite Micro source for ESP32 into lib/tensorflow_lite
# Usage: ./setup_tflite_micro.sh

set -e
FW_DIR="$(dirname "$0")"
LIB_DIR="$FW_DIR/lib/tensorflow_lite"
TMP_DIR=$(mktemp -d)

echo "Downloading TFLite Micro..."
git clone --depth 1 --single-branch \
  https://github.com/tensorflow/tflite-micro.git "$TMP_DIR/tflite-micro" 2>/dev/null

mkdir -p "$LIB_DIR"

# Copy only the needed subset for ESP32 inference
cp -r "$TMP_DIR/tflite-micro/tensorflow" "$LIB_DIR/tensorflow"
cp -r "$TMP_DIR/tflite-micro/third_party" "$LIB_DIR/third_party"

rm -rf "$TMP_DIR"
echo "TFLite Micro copied to $LIB_DIR"
echo "Size: $(du -sh "$LIB_DIR" | cut -f1)"
echo "Done."
