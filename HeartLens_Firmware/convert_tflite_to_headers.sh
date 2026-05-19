#!/bin/bash
# convert_tflite_to_headers.sh
# Converts trained .tflite files to C header arrays for the firmware.
# Usage: ./convert_tflite_to_headers.sh [training_dir] [firmware_dir]

TRAIN_DIR="${1:-$(dirname "$0")/heart-lens-training/models}"
FW_DIR="${2:-$(dirname "$0")/HeartLens_Firmware/models}"

for f in "$TRAIN_DIR"/*.tflite; do
  name=$(basename "$f" .tflite)
  out="$FW_DIR/${name}.h"
  xxd -i "$f" > "$out"
  echo "  $f → $out"
done
echo "Done."
