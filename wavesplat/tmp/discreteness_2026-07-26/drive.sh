#!/bin/bash
# Sequences the rest of the discreteness experiment once generation finishes.
cd /home/vexit/wavesplat/wavesplat
D=tmp/discreteness_2026-07-26

echo "[drive] waiting for 30 episodes ..."
until [ "$(ls $D/data/cell_a/*.h5 $D/data/cell_b/*.h5 $D/data/cell_c/*.h5 2>/dev/null | wc -l)" -ge 30 ]; do sleep 30; done
echo "[drive] 30 episodes present $(date +%T); waiting for generator processes to exit ..."
until [ "$(pgrep -fc 'gen_cells.py --cell')" = "0" ]; do sleep 20; done

echo "[drive] reindexing manifests $(date +%T)"
for c in a b c; do python3 $D/gen_cells.py --cell $c --reindex; done

echo "[drive] G-roundtrip $(date +%T)"
python3 $D/verify_roundtrip.py

echo "[drive] launching cond arm (GPU) + remaining WS arm (CPU) $(date +%T)"
python3 $D/run_cond.py > $D/cond_run.log 2>&1 &
CPID=$!
# WS: wait for the first pass to exit, then re-invoke for the cells that were missing
until [ "$(pgrep -fc 'run_ws.py')" = "0" ]; do sleep 20; done
OMP_NUM_THREADS=1 python3 $D/run_ws.py --workers 5 >> $D/ws_run.log 2>&1
echo "[drive] WS arm complete $(date +%T)"
wait $CPID
echo "[drive] cond arm complete $(date +%T)"

echo "[drive] secondary endpoints $(date +%T)"
python3 $D/run_secondary.py > $D/secondary.log 2>&1
echo "[drive] ALL RUNS COMPLETE $(date +%T)"
