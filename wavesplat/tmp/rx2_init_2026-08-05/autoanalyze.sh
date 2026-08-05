#!/bin/bash
cd /home/vexit/wavesplat/wavesplat/tmp/rx2_init_2026-08-05
for b in 1 2 3; do
  while ! grep -q "^DONE" sweep$b.log 2>/dev/null; do sleep 30; done
  python3 analyze_rx2.py > analysis$b.txt 2>&1
  python3 PRO_window_contrast.py > window$b.txt 2>&1
  echo "ANALYSIS$b READY"
done
