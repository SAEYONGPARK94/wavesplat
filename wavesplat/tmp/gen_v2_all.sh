#!/bin/bash
combos="BF2:1 BF3:1 BF3:2 BF3:5 BF4:1 BF4:5 BF4:15 BF4:20 BF4:30 BF5:15 BF5:30"
for c in $combos; do
  ss=${c%%:*}; nc=${c##*:}
  for seed in 0 1 2 3 4; do
    f="data/episode_${ss}_${nc}comp_seed${seed}_fixedT.h5"
    if [ -f "$f" ]; then echo "skip $f"; continue; fi
    python3 -m wavesim.pipeline --sea_state $ss --n_components $nc --seed $seed \
      --raydrop fixedT --n_pts 8000 --output_dir data > /dev/null 2>&1 \
      && echo "done $f" || echo "FAIL $f"
  done
done
echo V2_GENERATION_COMPLETE
