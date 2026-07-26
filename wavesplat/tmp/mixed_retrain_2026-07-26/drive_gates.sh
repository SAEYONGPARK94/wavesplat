#!/usr/bin/env bash
# MIXED-RETRAIN gate chain.  Runs only AFTER training has written
# results/direction3/beamform_cond_diffusion_mixed_v1.pt.  Every gate is run ONCE.
set -u
cd /home/vexit/wavesplat/wavesplat
D=tmp/mixed_retrain_2026-07-26
CKPT=results/direction3/beamform_cond_diffusion_mixed_v1.pt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "FATAL: $CKPT missing"; exit 1; }
echo "=== gate chain start $(date) ==="

echo "--- [1/5] G1+G2 GPU precompute (40 episodes x 5 diffusion seeds) ---"
python3 $D/precompute_new.py --ckpt "$CKPT" --out $D/atoms_cache_mixed_v1.jsonl \
  > $D/precompute.log 2>&1 || echo "precompute FAILED"
tail -2 $D/precompute.log

echo "--- [2/5] G3 continuum cell_c (GPU) ---"
python3 $D/run_cond_new.py --ckpt "$CKPT" --cells c --out $D/cond_runs_mixed_v1.jsonl \
  > $D/cond.log 2>&1 || echo "G3 FAILED"
tail -2 $D/cond.log

echo "--- [3/5] G4 real, 30 windows (GPU learned arm + CPU hand control) ---"
python3 $D/run_wp_real_new.py --ckpt "$CKPT" --out $D/wp_real_mixed_v1.json \
  > $D/wp_real.log 2>&1 || echo "G4-accuracy FAILED"
tail -2 $D/wp_real.log

echo "--- [4/5] G4 collapse, 18 harvested windows (GPU) ---"
python3 $D/run_collapse_new.py --ckpt "$CKPT" --out $D/collapse_learned_mixed_v1.json \
  > $D/collapse.log 2>&1 || echo "G4-collapse FAILED"
tail -2 $D/collapse.log

echo "--- [5/5] G1+G2 CPU tracker arms (crowded 16 + one-comp 12, 5 lanes) ---"
nice -n 10 python3 $D/run_arms_new.py --arms learned --episodes gates --workers 6 \
  --cache $D/atoms_cache_mixed_v1.jsonl --out $D/arm_runs_mixed_v1.jsonl \
  > $D/arms.log 2>&1 || echo "G1/G2 arms FAILED"
tail -2 $D/arms.log

echo "=== gate chain done $(date) ==="
