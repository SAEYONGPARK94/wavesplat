#!/bin/bash
# B3 / Rx-3 execution chain. PREREG.md frozen before this ran.
cd /home/vexit/wavesplat/wavesplat
export RX2_EPISODES=p1 WORKERS=8
echo "=== BATCH 0: reproduction control (AUTO, seeds 1518-1520, must bit-match rx2/B2) ==="
RX2_SEEDS=1518,1519,1520 RX2_OUT=tmp/B3_rx3_2026-08-06/repro.jsonl \
  python3 tmp/B3_rx3_2026-08-06/run_b3.py AUTO
echo "=== BATCH 1: main (6 cells, seeds 1521-1523) ==="
RX2_SEEDS=1521,1522,1523 RX2_OUT=tmp/B3_rx3_2026-08-06/results.jsonl \
  python3 tmp/B3_rx3_2026-08-06/run_b3.py AUTO,AUTO-DUP,AUTO-NOTRACK,AUTO-K6,AUTO-K7,AUTO-K4
echo "=== CHAIN DONE ==="
