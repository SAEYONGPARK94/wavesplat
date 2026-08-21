#!/bin/bash
# Rx-4 execution chain. Repro control FIRST (PREREG sec.6.1): if the copy is not
# bit-identical to B3 on AUTO x crowded4 x seeds 1521-1523, nothing else may be read.
set -x
cd /home/vexit/wavesplat/wavesplat
export WORKERS=8
export PYTHONUNBUFFERED=1
HERE=tmp/Rx4_tracker_init_2026-08-06

# lane 1: reproduction control (0 new seeds; EPISODES default == CROWDED4 == B3's crowded set)
RX2_SEEDS=1521,1522,1523 RX2_OUT=$HERE/repro.jsonl \
  python3 $HERE/run_rx4.py AUTO > $HERE/repro.log 2>&1

# lane 2: main batch, 7 cells x 24 episodes x seeds 1524/1525/1526
RX2_EPISODES=c16s8 \
  python3 $HERE/run_rx4.py AUTO,AUTO-DUP,AUTO-NOTRACK,T-PHIONLY,T-WONLY,T-EARLY20,T-K4REF \
  > $HERE/run.log 2>&1
echo CHAIN_DONE
