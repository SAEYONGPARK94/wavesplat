#!/usr/bin/env bash
# Parameterised gate chain.  Usage: drive_gates_vN.sh <version-suffix>   e.g. "v2"
set -u
cd /home/vexit/wavesplat/wavesplat
V=${1:?need version suffix, e.g. v2}
D=tmp/mixed_retrain_2026-07-26
CKPT=results/direction3/beamform_cond_diffusion_mixed_${V}.pt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "FATAL: $CKPT missing"; exit 1; }
echo "=== gate chain ${V} start $(date) ==="

echo "--- [1/5] G1+G2 GPU precompute ---"
python3 $D/precompute_new.py --ckpt "$CKPT" --out $D/atoms_cache_mixed_${V}.jsonl \
  > $D/precompute_${V}.log 2>&1 || echo "precompute FAILED"; tail -1 $D/precompute_${V}.log

echo "--- [2/5] G3 continuum cell_c ---"
python3 $D/run_cond_new.py --ckpt "$CKPT" --cells c --out $D/cond_runs_mixed_${V}.jsonl \
  > $D/cond_${V}.log 2>&1 || echo "G3 FAILED"; tail -1 $D/cond_${V}.log

echo "--- [3/5] G4 real, 30 windows ---"
python3 $D/run_wp_real_new.py --ckpt "$CKPT" --out $D/wp_real_mixed_${V}.json \
  > $D/wp_real_${V}.log 2>&1 || echo "G4-accuracy FAILED"; tail -1 $D/wp_real_${V}.log

echo "--- [4/5] G4 collapse, 18 windows ---"
python3 $D/run_collapse_new.py --ckpt "$CKPT" --out $D/collapse_learned_mixed_${V}.json \
  > $D/collapse_${V}.log 2>&1 || echo "G4-collapse FAILED"; tail -1 $D/collapse_${V}.log

echo "--- [5/5] G1+G2 CPU tracker arms ---"
nice -n 10 python3 $D/run_arms_new.py --arms learned --episodes gates --workers 6 \
  --cache $D/atoms_cache_mixed_${V}.jsonl --out $D/arm_runs_mixed_${V}.jsonl \
  > $D/arms_${V}.log 2>&1 || echo "G1/G2 arms FAILED"; tail -1 $D/arms_${V}.log

echo "=== gate chain ${V} done $(date) ==="
