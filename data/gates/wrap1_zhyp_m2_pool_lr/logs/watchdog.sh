#!/bin/bash
# Heartbeat + cross-fold memory check for one wrap1_zhyp_m2_pool_lr arm run.
# usage: watchdog.sh [ARM]     (default lr_1e-4, the arm this script was first used for)
# Not part of the sealed protocol -- operational monitoring only. Exits once the
# run's DONE marker appears. Every line is timestamped so the file is checkable
# at any time without touching the running process.
set -u
ARM="${1:-lr_1e-4}"
BASE="/workspace/data/gates/wrap1_zhyp_m2_pool_lr"
OUT_DIR="$BASE/$ARM"
DONE_MARKER="$BASE/logs/DONE_$ARM"
RUN_LOG="$BASE/logs/${ARM}_run.log"
HEARTBEAT="$BASE/logs/heartbeat_${ARM}.log"
[ "$ARM" = "lr_1e-4" ] && HEARTBEAT="$BASE/logs/heartbeat.log"   # legacy filename kept for the first arm

prev_fold_count=0
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  fold_count=$(ls "$OUT_DIR" 2>/dev/null | grep -c '\.json$')
  last_step=$(grep -a 'step=' "$RUN_LOG" 2>/dev/null | grep -av 'FINAL' | tail -1)
  echo "$ts mem=${mem}MiB util=${util}% folds_done=${fold_count} last='${last_step}'" >> "$HEARTBEAT"
  if [ "$fold_count" -gt "$prev_fold_count" ]; then
    echo "$ts *** FOLD_BOUNDARY ${prev_fold_count}->${fold_count}: mem=${mem}MiB (compare to prior heartbeat lines for drift) ***" >> "$HEARTBEAT"
    prev_fold_count=$fold_count
  fi
  if [ -f "$DONE_MARKER" ]; then
    echo "$ts watchdog exiting: run DONE marker found" >> "$HEARTBEAT"
    break
  fi
  sleep 300
done
