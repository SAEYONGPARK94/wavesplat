#!/bin/bash
# Sequences the aperture arm once its generation finishes.
cd /home/vexit/wavesplat/wavesplat
D=tmp/discreteness_2026-07-26

echo "[drive] waiting for 20 aperture episodes ..."
until [ "$(ls $D/data/cell_abig/*.h5 $D/data/cell_cbig/*.h5 2>/dev/null | wc -l)" -ge 20 ]; do sleep 30; done
until [ "$(pgrep -fc 'gen_cells.py --cell')" = "0" ]; do sleep 20; done
echo "[drive] generation done $(date +%T); reindexing"
for c in abig cbig; do python3 $D/gen_cells.py --cell $c --reindex; done

echo "[drive] cond arm (GPU) + WS arm (4 CPU workers) $(date +%T)"
python3 $D/run_cond_aperture.py > $D/cond_run_aperture.log 2>&1 &
CPID=$!
OMP_NUM_THREADS=1 python3 $D/run_ws_aperture.py --workers 4 > $D/ws_run_aperture.log 2>&1
echo "[drive] WS done $(date +%T)"
wait $CPID
echo "[drive] cond done $(date +%T)"

echo "[drive] secondary $(date +%T)"
python3 $D/run_secondary_aperture.py > $D/secondary_aperture.log 2>&1
echo "[drive] APERTURE RUNS COMPLETE $(date +%T)"
