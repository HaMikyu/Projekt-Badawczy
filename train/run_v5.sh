#!/bin/bash
# Trening wariantow v5 po kolei + ocena na dev (bez testow esejowych i finalnego).
#   bash train/run_v5.sh [a d]
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for x in ${@:-a d}; do
  run="train/runs/system-one-v5$x"
  mkdir -p "$run"
  "$PY" -u train/train_system_one_v5.py "$x" > "$run/train.log" 2>&1
  code=$?
  tr '\r' '\n' < "$run/train.log" | grep -E "tokenizacja|gotowe|Error|Traceback|out of memory" | tail -5
  if [ $code -ne 0 ]; then echo "[$x] TRENING NIEUDANY (kod $code)"; continue; fi
  "$PY" train/eval_dev.py "$run/export" > "$run/eval_dev.log" 2>&1
  echo "[$x] eval_dev zapisany: $run/eval_dev.log"
done
