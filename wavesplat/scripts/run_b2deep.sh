#!/bin/bash
# B-2 DEEP driver: 4 sequential runs (T-s0 -> L-s0 -> T-s1 -> L-s1), 12k each.
cd /home/vexit/wavesplat/wavesplat
set -e
run () {  # obj seed ckpt
  echo "======== $(date +%H:%M:%S) START $1 s$2 -> $3 ========"
  python3 scripts/d3_b2deep_train.py --objective "$1" --seed "$2" \
      --steps 12000 --val_every 500 --auto_extend 1 --ckpt "$3"
  echo "======== $(date +%H:%M:%S) DONE $1 s$2 ========"
}
run task  0 results/direction3/b2deep_T_s0.pt
run resid 0 results/direction3/b2deep_L_s0.pt
run task  1 results/direction3/b2deep_T_s1.pt
run resid 1 results/direction3/b2deep_L_s1.pt
echo "======== $(date +%H:%M:%S) ALL 4 RUNS COMPLETE ========"
