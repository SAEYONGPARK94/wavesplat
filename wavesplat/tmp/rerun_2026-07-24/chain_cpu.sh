#!/bin/bash
# CPU jobs that must run WITHOUT the GPU chain contending (ETKF/GPR became 100x slower under
# load-34 contention). Waits for the GPU chain to finish, then runs them on a free machine.
cd /home/vexit/wavesplat/wavesplat
D=tmp/rerun_2026-07-24
until grep -q "ALL GPU CHAINS DONE" $D/chain_gpu.out 2>/dev/null; do sleep 120; done
echo "gpu chain done, 100ep tuned ETKF $(date)"
python3 $D/etkf_100ep_tuned.py > $D/etkf_100ep.log 2>&1
echo "etkf 100ep done, GPR $(date)"
python3 $D/gprjobs.py           > $D/gpr_runs.log 2>&1
echo "ALL CPU CHAINS DONE $(date)"
