#!/usr/bin/env bash
# keep-here progress watcher — run anytime, foreground-friendly
ROOT="/home/touhid/heartlens"
TR="$ROOT/heart-lens-training"
MARK="$ROOT/.markers"
echo "=== PIPELINE ($(date '+%F %R')) ==="
TOTAL=10
for i in 1 2 3 4 5 6 7 8 9 10; do
  name=$(ls $MARK/*.done 2>/dev/null | wc -l)
  break
done
if [ -f "$TR/results/pipeline_progress.txt" ]; then cat "$TR/results/pipeline_progress.txt"; fi
if [ -f "$TR/results/progress.txt" ]; then echo "--- group_kfold progress ---"; cat "$TR/results/progress.txt"; fi
echo ""
echo "Markers:"
ls -1 $MARK/*.done 2>/dev/null | xargs -I{} basename {} | tr '\n' ' '; echo ""
echo ""
echo "Ckpts:"
if [ -d "$TR/results/group_kfold_ckpt" ]; then
  cnt=$(ls "$TR/results/group_kfold_ckpt"/*.json 2>/dev/null | wc -l)
  tot=20  # cnn,tcn x 5x2
  pct=$(( cnt * 100 / tot ))
  echo "  group_kfold_ckpt: $cnt/$tot ($pct%)"
  ls "$TR/results/group_kfold_ckpt"/ 2>/dev/null | head -20 | tr '\n' ' '; echo ""
fi
if [ -d "$TR/results/apb_ckpt" ]; then
  cnt=$(ls "$TR/results/apb_ckpt"/*.json 2>/dev/null | wc -l)
  echo "  apb_ckpt: $cnt/8 ($((cnt*100/8))%)"
fi
if [ -d "$TR/results/paired_quant" ]; then
  cnt=$(ls "$TR/results/paired_quant"/*.json 2>/dev/null | wc -l)
  echo "  paired_quant: $cnt/20 ($((cnt*100/20))%)"
fi
echo ""
echo "Last log tail (group_kfold_all):"
tail -n 5 "$ROOT/train_logs/group_kfold_all.log" 2>/dev/null | tr -d '\r' | tail -n 5
