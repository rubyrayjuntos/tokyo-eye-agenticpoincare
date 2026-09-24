#!/bin/bash
# Heartbeat + cross-fold memory check for the lr_1e-4 wrap1_zhyp_m2_pool_lr run.
# Not part of the sealed protocol -- pure operational monitoring. Exits once
# the run's own DONE marker appears. Every line is timestamped so this file
# is checkable at any time without touching the running process.
set -u
OUT_DIR="/workspace/data/gates/wrap1_zhyp_m2_pool_lr/lr_1e-4"
DONE_MARKER="/workspace/data/gates/wrap1_zhyp_m2_pool_lr/logs/DONE_lr_1e-4"
RUN_LOG="/workspace/data/gates/wrap1_zhyp_m2_pool_lr/logs/lr_1e-4_run.log"
HEARTBEAT="/workspace/data/gates/wrap1_zhyp_m2_pool_lr/logs/heartbeat.log"

prev_fold_count=0
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  fold_count=$(ls "$OUT_DIR" 2>/dev/null | grep -c '\.json$')
  last_step=$(grep -a 'step=' "$RUN_LOG" 2>/dev/null | tail -1)
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
