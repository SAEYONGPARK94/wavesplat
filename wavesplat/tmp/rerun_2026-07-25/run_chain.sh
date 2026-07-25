#!/bin/bash
# Sequential chain (<=4 worker processes at a time): GPR-3 (Task 2 arm) then WS (Task 3+4).
cd /home/vexit/wavesplat/wavesplat
export WORKERS=4
# wait for batchA (Task 1 baselines) to finish first, so we never exceed 4 workers
until grep -q DONE tmp/rerun_2026-07-25/batchA.log 2>/dev/null; do sleep 5; done
echo "=== batchA observed DONE $(date) ===" >> tmp/rerun_2026-07-25/chain.log
echo "=== batchA2 GPR-3 start $(date) ===" >> tmp/rerun_2026-07-25/chain.log
nice -n 10 python3 tmp/rerun_2026-07-25/batchA2_gpr3.py >> tmp/rerun_2026-07-25/batchA2.log 2>&1
echo "=== batchA2 done $(date) ===" >> tmp/rerun_2026-07-25/chain.log
echo "=== batchB WS start $(date) ===" >> tmp/rerun_2026-07-25/chain.log
nice -n 10 python3 tmp/rerun_2026-07-25/batchB_ws.py >> tmp/rerun_2026-07-25/batchB.log 2>&1
echo "=== batchB done $(date) ===" >> tmp/rerun_2026-07-25/chain.log
