#!/usr/bin/env bash
set -u
cd /home/vexit/wavesplat/wavesplat
while pgrep -f "gen_sparse.py" >/dev/null; do sleep 20; done
python3 tmp/S1_indep_sparse_2026-08-08/gen_crowded.py
