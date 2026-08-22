#!/usr/bin/env bash
#
# HeartLens 12h hardware-only runner — resumable & reboot-safe (FOREGROUND).
#
#   ./run_experiments.sh
#
# - All steps run in FOREGROUND with `tee` so you can watch.
# - Completed steps marked in .markers/*.done and skipped on re-run.
# - Per-fold checkpoints inside steps mean a reboot only loses the in-flight FOLD, not the whole step.
# - Before retrying a step, its FINAL outputs are checked via `need $SRC`; partial .tmp files are ignored.
# - Every Python script uses unbuffered output (PYTHONUNBUFFERED=1) so train_logs/* shows progress after reboot.
# - Stops on first failure so you see the error; just re-run the same command to resume.
set -uo pipefail

ROOT="/home/touhid/heartlens"
TR="$ROOT/heart-lens-training"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
MARK="$ROOT/.markers"
LOGS="$ROOT/train_logs"
export PYTHONUNBUFFERED=1
mkdir -p "$MARK" "$LOGS"

TOTAL=11
N=0

bump() { N=$((N + 1)); }

need() {
  for f in "$@"; do
    if [ ! -s "$f" ]; then
      echo "!! output missing/empty: $f" >&2
      return 1
    fi
  done
}

step() {
  local name="$1"; shift
  local cwd="$1"; shift
  [ "$1" = "--" ] && shift
  local marker="$MARK/$name.done"
  bump
  # pipeline % — keep-here visible
  local pct=$(( N * 100 / TOTAL ))
  if [ -f "$marker" ]; then
    printf '  -- [%2d/%d] %3d%% skip  %s (done)\n' "$N" "$TOTAL" "$pct" "$name"
    return 0
  fi
  printf '=====> [%2d/%d] %3d%% RUN  %s   %s\n' "$N" "$TOTAL" "$pct" "$name" "$(date '+%F %R')"
  set -o pipefail
  # foreground: no nohup, directly tee to terminal + log
  ( cd "$cwd" && "$@" 2>&1 | tee "$LOGS/$name.log" )
  local rc=$?
  if [ $rc -ne 0 ]; then
    printf '!! FAILED: %s (exit %d) — log: train_logs/%s.log\n' "$name" "$rc" "$name"
    return 1
  fi
  # shellcheck disable=SC2086
  if ! need $SRC; then
    printf '!! FAILED: %s (output check) — log: train_logs/%s.log\n' "$name" "$name"
    return 1
  fi
  touch "$marker"
  local pct_done=$(( (N) * 100 / TOTAL ))
  printf '=====> [%2d/%d] %3d%% DONE %s\n' "$N" "$TOTAL" "$pct_done" "$name"
  # keep-here pipeline progress file
  echo "PIPELINE $pct_done% ($N/$TOTAL) $name done @ $(date '+%F %R')" > "$TR/results/pipeline_progress.txt"
}

echo "HeartLens 12h hardware runner (foreground, resumable)"
echo "Reboot? Just re-run: ./run_experiments.sh  — it skips done steps, resumes folds."
echo ""

# 1. venv + deps (Python 3.12 via uv; TF 2.x supports 3.12)
SRC="$VENV/bin/python"
step venv "$ROOT" -- bash -c "
  if [ ! -x '$PY' ]; then
    rm -rf '$VENV'
    uv venv --python 3.12 '$VENV'
    uv pip install --python '$PY' tensorflow-cpu wfdb fpdf2 scikit-learn scipy matplotlib
  else
    echo \"venv exists: $PY\"
  fi
" || exit 1

# 2. Download MIT-BIH (crash-safe: loader rmtrees partial dirs)
SRC="$TR/mitdb/100.dat"
step mitdb "$TR" -- bash -c "
  if [ ! -s '$TR/mitdb/100.dat' ]; then
    rm -rf '$TR/mitdb'
    '$PY' -c \"from data_loader import download_mitdb; download_mitdb('$TR/mitdb')\"
  else
    echo \"mitdb exists\"
  fi
" || exit 1

# 3. Freeze folds (30s) — identical splits for every arch
SRC="$TR/results/folds_5x2.json"
step freeze_folds "$ROOT" -- "$PY" scripts/freeze_folds.py --folds 5 --seeds 0,1 --max-per-class 3000 || exit 1

# 4. 4-arch grouped CV (5 folds x 2 seeds = 20 folds per arch)
#    Resume: per-fold ckpts in results/group_kfold_ckpt/*.json — reboot loses only one fold
#    ponytail: default cnn,tcn for 12h; add lstm,gru if time allows (change --types)
SRC="$TR/results/group_kfold_all.json"
step group_kfold_all "$TR" -- "$PY" group_kfold_eval_v2.py --types cnn,tcn --folds 5 --seeds 0,1 --epochs 30 --folds-file results/folds_5x2.json || exit 1

# 5. Optional full 4-arch (runs only if you have time — same ckpt resume)
#    This step is SKIPPED if group_kfold_all already has lstm/gru ckpts; delete marker to force.
SRC="$TR/results/group_kfold_lstm.json $TR/results/group_kfold_gru.json"
step group_kfold_extra "$TR" -- "$PY" group_kfold_eval_v2.py --types lstm,gru --folds 5 --seeds 0,1 --epochs 30 --folds-file results/folds_5x2.json || {
  echo "group_kfold_extra failed or skipped — continuing (ponytail: lstm/gru are not deployable anyway)"
  rm -f "$MARK/group_kfold_extra.done"
}

# 6. Noise robustness (inference only, ~5 min)
SRC="$TR/results/noise_robustness.json"
step noise_robustness "$TR" -- bash -c "
  if [ ! -s '$TR/results/noise_robustness.json' ]; then
    '$PY' evaluate_noise_robustness.py --epochs 30
  else
    echo \"noise_robustness exists\"
  fi
" || exit 1

# 7. APB ablation — FULL 4-arch × 4 strategies = 16 trainings (was 8) — brutal C2
SRC="$TR/results/apb_ablation.json"
step apb_ablation "$ROOT" -- "$PY" scripts/apb_ablation.py --types cnn,tcn,lstm,gru --epochs 30 || exit 1

# 8. Paired quant — FULL 4-arch per-fold FP32 vs INT8 — brutal C4
SRC="$TR/results/paired_quant.json"
step paired_quant "$ROOT" -- "$PY" scripts/paired_quant.py --types cnn,tcn,lstm,gru --epochs 30 || exit 1

# 9. Calibration (fits T on val, writes Config.h, ~3 min)
SRC="$TR/results/calibration.json"
step calibrate "$TR" -- "$PY" calibrate.py --epochs 30 --write-config || exit 1

# 10. External validation SVDB (fix: N/V-macro only, ~4 min)
SRC="$TR/results/external_validation.json"
step external_validation "$TR" -- bash -c "
  if [ -s '$TR/results/external_validation.json' ] && grep -q 'svdb_macro_nv' '$TR/results/external_validation.json'; then
    echo \"external_validation exists (N/V-macro fixed)\"
  else
    rm -rf '$TR/svdb'
    '$PY' external_validation.py --skip-afdb --epochs 30
    '$PY' /home/touhid/heartlens/scripts/fix_svdb_metric.py
  fi
" || exit 1

# 11. Brutal-review synthesis — per-artifact, SQI, calibration diagram, deployment master (all inference-only, fast)
SRC="$TR/results/per_artifact_noise.json $TR/results/sqi_ablation.json $TR/results/reliability_diagram.png $TR/results/deployment_master.json"
step brutal_synthesis "$ROOT" -- bash -c "
  '$PY' scripts/per_artifact_noise.py --epochs 10
  '$PY' scripts/sqi_ablation.py
  '$PY' scripts/generate_calibration_diagram.py
  '$PY' scripts/generate_deployment_table.py
" || exit 1

echo ""
echo "ALL STEPS COMPLETE. (Foreground, resumable via .markers/*.done)"
echo "Results: $TR/results/   Models: $TR/models/   Logs: $LOGS/"
echo "Next (needs S3 board, foreground):"
echo "  pio run -e esp32-s3-devkitc-1 -t upload && pio device monitor  # BENCHMARK_MODE"
echo "  python3 hw_eval/replay_drive.py --port /dev/ttyACM0 --model heart-lens-training/models/robust_classifier_int8.tflite --n 300"
