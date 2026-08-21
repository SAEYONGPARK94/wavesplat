#!/bin/bash
set -x
cd /home/vexit/wavesplat/wavesplat
export WORKERS=8 PYTHONUNBUFFERED=1 RX5_LR_W_MATCHED=1e-9 RX2_SEEDS=1539,1540,1541
HERE=tmp/EB_instrument_2026-08-07
CELLS=AUTO,AUTO-DUP,AUTO-LR10,AUTO-LR100,AUTO-LR1000,AUTO-NOTRACK
# lane 1 (PRIMARY): fresh 16
RX2_EPISODES=fresh16 RX2_OUT=$HERE/results_lane1.jsonl python3 $HERE/run_eb.py $CELLS > $HERE/run_lane1.log 2>&1
# lane 2 (co-report only, never pooled): the existing 32
RX2_EPISODES=indep32 RX2_OUT=$HERE/results_lane2.jsonl python3 $HERE/run_eb.py $CELLS > $HERE/run_lane2.log 2>&1
echo EB_CHAIN_DONE
