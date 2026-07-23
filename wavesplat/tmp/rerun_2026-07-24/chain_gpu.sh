#!/bin/bash
cd /home/vexit/wavesplat/wavesplat
D=tmp/rerun_2026-07-24
# wait for ep100 (the headline) to finish
until [ "$(grep -c '"nRMSE"' $D/ep100_runs.jsonl 2>/dev/null)" -ge 500 ]; do sleep 60; done
echo "ep100 done, starting flag sweep $(date)"
WS_WORKERS=5 python3 $D/flagjobs.py           > $D/flag.log 2>&1
echo "flag done, task13 $(date)"
WS_WORKERS=5 python3 $D/task13_init_vs_track.py > $D/task13.log 2>&1
echo "task13 done, hyper sweep $(date)"
WS_WORKERS=5 python3 $D/sweepjobs.py          > $D/sweep.log 2>&1
echo "sweep done, ablation-prod $(date)"
WS_WORKERS=5 python3 $D/ablation_prod_init.py > $D/ablprod.log 2>&1
echo "ALL GPU CHAINS DONE $(date)"
