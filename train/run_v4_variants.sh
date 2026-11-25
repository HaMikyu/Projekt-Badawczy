#!/bin/bash
# Trening i ocena na dev wariantow v4 po kolei (bez finalnego testu).
#   bash train/run_v4_variants.sh [a b c]
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for v in "${@:-a b c}"; do
  for x in $v; do
    run="train/runs/system-one-v4$x"
    mkdir -p "$run"
    "$PY" -u train/train_system_one_v4.py "$x" > "$run/train.log" 2>&1
    code=$?
    tail -c 2000 "$run/train.log" | tr '\r' '\n' | grep -E "tokenizacja|gotowe|Error|Traceback" || true
    if [ $code -ne 0 ]; then echo "[$x] TRENING NIEUDANY (kod $code)"; exit $code; fi
    "$PY" train/eval_dev.py "$run/export" > "$run/eval_dev.log" 2>&1
    echo "[$x] eval_dev zapisany: $run/eval_dev.log"
  done
done
