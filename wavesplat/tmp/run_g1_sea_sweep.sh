#!/usr/bin/env bash
# G1b: hold everything fixed except the number of DISTINCT training seas.
#
# W7 (13 real records)   leave-one-record-out R2 = -0.603
# G1 (1600 synth seas)   leave-sea-out       R2 = +0.53
#
# Those two differ in more than sea count -- label, noise, sensor. This sweep
# changes ONLY the sea count, on the same synthetic data, same architecture,
# same 20k steps (G1's probe R2 was flat from 10k, so steps are not the axis).
#
# If R2 collapses at 13 seas and climbs with diversity, the binding constraint
# is how many distinct seas you have seen, not how long you trained.
set -e
cd "$(dirname "$0")/.."
for S in 13 25 50 100 400 1600; do
  echo "=== train_seas ${S} ==="
  python3 -u scripts/d3_g1_synth_ssl.py --seas 2000 --held 400 \
      --train_seas "${S}" --steps 20000 --tag "_ns${S}" 2>&1 \
    | grep -E "^  step +20000|^  plug-in|^  random-init|^  SSL|^  PASS|^  FAIL"
done
