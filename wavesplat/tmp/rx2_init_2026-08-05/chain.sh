#!/bin/bash
cd /home/vexit/wavesplat/wavesplat
H=tmp/rx2_init_2026-08-05
# wait for batch 1
while ! grep -q "^DONE" $H/sweep1.log 2>/dev/null; do sleep 20; done
# batch 2 : corrected denominator + anchor + stack + the ACTUAL shipped default
W=8; pgrep -f run_e2.py >/dev/null && W=5
WORKERS=$W python3 $H/run_rx2b.py X-BASE3,ORACLE-K,G1,G1P3,AUTO > $H/sweep2.log 2>&1
# batch 3 : predicted-null axis W + ladder axis S + remaining stack
W=10; pgrep -f run_e2.py >/dev/null && W=6
WORKERS=$W python3 $H/run_rx2b.py X-BASE4,W1,W2,W3,S1,S2,G1P1 > $H/sweep3.log 2>&1
echo ALLDONE
