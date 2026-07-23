#!/bin/bash
# Regenerate the fixedT episodes A-7 v2 needs, at the Nyquist-adequate mesh.
#
# Why: A-7 v2 (scripts/d3_a7v2_refine.py) is gated on BF2 1comp fixedT x5 and calibrates its
# kp-invariant detector on BF3/BF4 1comp-vs-5comp fixedT. Those files exist only in `data/`,
# which the main regeneration deliberately excluded (REGEN_SPEC: fixedT/dir180/BF6-7 are not
# part of the 100-episode paired set). Running A-7 against old fixedT files would defeat the
# entire purpose of the retry, and calibrating on old BF3/BF4 while judging on new BF2 would
# be worse still -- so the calibration set is regenerated too.
#
# Mesh 200x167 matches the main regeneration (dx = 1.206 m). Same seeds -> same components.
set -u
OUT=data_finemesh_fixedT_2026-07-23
mkdir -p "$OUT"
combos="BF2:1 BF3:1 BF3:5 BF4:1 BF4:5"     # gate set + calibration sets only
for c in $combos; do
  ss=${c%%:*}; nc=${c##*:}
  for seed in 0 1 2 3 4; do
    f="$OUT/episode_${ss}_${nc}comp_seed${seed}_fixedT.h5"
    if [ -f "$f" ]; then echo "skip $f"; continue; fi
    python3 -m wavesim.pipeline --sea_state "$ss" --n_components "$nc" --seed "$seed" \
      --raydrop fixedT --n_pts 8000 --output_dir "$OUT" \
      --mesh_nx 200 --mesh_ny 167 > /dev/null 2>&1 \
      && echo "done $f" || echo "FAIL $f"
  done
done
echo "TOTAL: $(ls "$OUT"/*.h5 2>/dev/null | wc -l) files"
