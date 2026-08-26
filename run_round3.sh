#!/usr/bin/env bash
# Round-3 experiment driver — resumable after restart.
# Each stage writes .markers/round3/<stage>.done on success; re-running skips done stages.
# Individual training stages are additionally per-fold resumable inside Python.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
MK=.markers/round3
mkdir -p "$MK"
LOG=heart-lens-training/results/round3.log
mkdir -p "$(dirname "$LOG")"
touch "$LOG"

stage() {  # stage <name> <cmd...>
  local name="$1"; shift
  if [ -f "$MK/$name.done" ]; then echo "[driver] $name already done, skip"; return 0; fi
  echo "[driver] === stage $name start $(date) ===" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    touch "$MK/$name.done"
    echo "[driver] === stage $name DONE $(date) ===" | tee -a "$LOG"
  else
    echo "[driver] !!! stage $name FAILED (see $LOG) $(date)" | tee -a "$LOG"
    return 1
  fi
}

stage s1_baseline_cv $PY heart-lens-training/group_kfold_eval_v2.py \
  --types cnn,tcn --class-weights none --tag _baseline --epochs 30
stage s2_qat_cv      $PY scripts/round3_qat_cv.py --types cnn --epochs 30 --qat-epochs 3
stage s3_int8_calib  $PY scripts/round3_int8_calibration.py
stage s4_analysis    $PY scripts/round3_analysis.py

echo "[driver] all compute stages complete $(date)" | tee -a "$LOG"
