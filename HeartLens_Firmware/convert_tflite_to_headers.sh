#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="${1:-$SCRIPT_DIR/heart-lens-training/models}"
FW_DIR="${2:-$SCRIPT_DIR/HeartLens_Firmware/models}"

if [ ! -d "$TRAIN_DIR" ]; then
  echo "ERROR: Source directory not found: $TRAIN_DIR" >&2
  exit 1
fi

mkdir -p "$FW_DIR"

for f in "$TRAIN_DIR"/*.tflite; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .tflite)
  out="$FW_DIR/${name}.h"

  array_name="$(echo "$name" | sed 's/[^a-zA-Z0-9_]/_/g')"

  {
    printf '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
    # xxd names the symbol after the input path — run from the file's dir
    # so the symbol is just <basename>_tflite, not the absolute path.
    (cd "$(dirname "$f")" && xxd -i "$(basename "$f")") | \
      sed 's/^unsigned /unsigned const /'
    printf '\n#ifdef __cplusplus\n}\n#endif\n'
  } > "$out"

  echo "  $f → $out"
done

echo "Done."
