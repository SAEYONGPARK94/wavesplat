#!/bin/bash
# Wait for the fixedT gap-batch worker(s), then run the extras
# (BF6/BF7 + the two stray uniform episodes) at the same 200x167 mesh.
set -u
cd /home/vexit/wavesplat/wavesplat
for PID in "$@"; do
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
done
echo "[chain] fixedT workers exited, starting extras" >> tmp/regen_gap_extras.log
python3 tmp/regen_gap_fixedT.py \
  --list tmp/regen_gap_extras_todo.txt \
  --output_dir data_finemesh_extras_2026-07-24 \
  --log tmp/regen_gap_extras.log >> tmp/regen_gap_extras.stdout 2>&1
echo "[chain] extras done" >> tmp/regen_gap_extras.log
