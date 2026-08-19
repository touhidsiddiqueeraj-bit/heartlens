#!/usr/bin/env bash
#
# HeartLens local training runner — resumable & reboot-safe (foreground).
#
#   ./run_experiments.sh
#
# - Runs steps in order. Completed steps are marked in .markers/ and skipped.
# - A reboot/kill only loses the in-flight step; re-run this same command and
#   it continues from the first undone step.
# - Before retrying an interrupted step, its outputs are deleted so a partial
#   model/JSON can never be silently reused.
# - Step output is streamed to the terminal AND teed to train_logs/<step>.log
# - Stops on first failure so you can see the error; markers persist.
set -uo pipefail

ROOT="/home/touhid/heartlens"
TR="$ROOT/heart-lens-training"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
MARK="$ROOT/.markers"
LOGS="$ROOT/train_logs"
mkdir -p "$MARK" "$LOGS"

TOTAL=14
N=0

bump() { N=$((N + 1)); }

need() {
  # need <file>... — each must exist and be non-empty
  for f in "$@"; do
    if [ ! -s "$f" ]; then
      echo "!! output missing/empty: $f" >&2
      return 1
    fi
  done
}

step() {
  # step <name> <cwd> -- <cmd...>
  local name="$1"; shift
  local cwd="$1"; shift
  [ "$1" = "--" ] && shift
  local marker="$MARK/$name.done"
  bump
  if [ -f "$marker" ]; then
    printf '  -- [%2d/%d] skip  %s (done)\n' "$N" "$TOTAL" "$name"
    return 0
  fi
  printf '=====> [%2d/%d] RUN  %s   %s\n' "$N" "$TOTAL" "$name" "$(date '+%F %R')"
  set -o pipefail
  ( cd "$cwd" && "$@" 2>&1 | tee "$LOGS/$name.log" )
  local rc=$?
  if [ $rc -ne 0 ]; then
    printf '!! FAILED: %s (exit %d) — log: train_logs/%s.log\n' "$name" "$rc" "$name"
    return 1
  fi
  # shellcheck disable=SC2086  # SRC is an intentional space-separated file list
  if ! need $SRC; then
    printf '!! FAILED: %s (output check) — log: train_logs/%s.log\n' "$name" "$name"
    return 1
  fi
  touch "$marker"
  printf '=====> [%2d/%d] DONE %s\n' "$N" "$TOTAL" "$name"
}

# Each step re-executes SRC (the output check target) via $SRC after running.
# Note: steps are sequential; a failing step stops the run for inspection.

# 1. venv + deps (Python 3.12 via uv; TF supports 3.12, not 3.14)
SRC="$VENV/bin/python"
step venv "$ROOT" -- bash -c "
  rm -rf '$VENV'
  uv venv --python 3.12 '$VENV'
  uv pip install --python '$PY' tensorflow-cpu wfdb fpdf2 scikit-learn scipy matplotlib
" || exit 1

# 2. Download MIT-BIH (crash-safe: loader rmtrees partial dirs)
SRC="$TR/mitdb/100.dat"
step mitdb "$TR" -- bash -c "
  rm -rf '$TR/mitdb'
  '$PY' -c \"from data_loader import download_mitdb; download_mitdb('$TR/mitdb')\"
" || exit 1

# 3-4. auto_train denoiser, then classifier (PDF report; distinct markers)
SRC="$ROOT/auto_train_output/training_report.pdf"
step auto_train_denoiser "$ROOT" -- "$PY" auto_train.py \
  --data-dir heart-lens-training/mitdb --skip-classifier --epochs 30 --max-per-class 3000 || exit 1
step auto_train_classifier "$ROOT" -- "$PY" auto_train.py \
  --data-dir heart-lens-training/mitdb --skip-denoiser --epochs 30 --max-per-class 3000 || exit 1

# 5. Exp 1: grouped patient-level CV (5 folds x 3 seeds)
SRC="$TR/results/group_kfold.json"
step group_kfold "$TR" -- bash -c "
  rm -f '$TR/results/group_kfold.json'
  '$PY' group_kfold_eval.py --folds 5 --seeds 0,1,2 --epochs 30
" || exit 1

# 6. Exp 2: noise robustness (also trains + saves robust_*.keras)
SRC="$TR/results/noise_robustness.json $TR/models/robust_classifier.keras $TR/models/robust_denoiser.keras"
step noise_robustness "$TR" -- bash -c "
  rm -f '$TR/results/noise_robustness.json' '$TR/models/robust_classifier.keras' '$TR/models/robust_denoiser.keras'
  '$PY' evaluate_noise_robustness.py --epochs 30
" || exit 1

# 7-10. Exp 4: compare architectures, one step each (per-arch JSON outputs)
for ARCH in cnn lstm gru tcn; do
  SRC="$TR/results/model_comparison_$ARCH.json"
  step "compare_$ARCH" "$TR" -- bash -c "
    rm -f '$TR/results/model_comparison_$ARCH.json' '$TR/results/model_comparison_$ARCH.csv' '$TR/results/model_comparison_$ARCH.png'
    '$PY' compare_models.py --types $ARCH --suffix _$ARCH --epochs 30
  " || exit 1
done

# 11. Merge per-arch comparison rows
SRC="$TR/results/model_comparison.json"
step merge_comparison "$TR" -- "$PY" merge_comparison.py || exit 1

# 12. Calibration -> writes CALIB_TEMPERATURE into firmware Config.h
SRC="$TR/results/calibration.json"
step calibrate "$TR" -- bash -c "
  rm -f '$TR/results/calibration.json'
  '$PY' calibrate.py --epochs 30 --write-config
" || exit 1

# 13. Exp 3: external generalization mitdb -> SVDB (AFDB skipped)
SRC="$TR/results/external_validation.json"
step external_validation "$TR" -- bash -c "
  rm -f '$TR/results/external_validation.json'
  rm -rf '$TR/svdb'
  '$PY' external_validation.py --skip-afdb --epochs 30
" || exit 1

# 14. Export int8 firmware models (needs step 6's robust_*.keras)
SRC="$TR/models/robust_classifier_int8.tflite $TR/models/robust_denoiser_int8.tflite"
step export_firmware "$TR" -- bash -c "
  rm -f '$TR/models/robust_classifier_int8.tflite' '$TR/models/robust_denoiser_int8.tflite'
  '$PY' export_firmware_models.py
" || exit 1

echo ""
echo "ALL STEPS COMPLETE."
echo "Results: $TR/results/   Models: $TR/models/   Report: $ROOT/auto_train_output/training_report.pdf"
