#!/bin/bash
# GPU-phase chain: gate the atoms cache, then the three tracker arms (learned, v2 control,
# auto live).  Frozen protocol: tmp/heldout40_2026-07-26/PREREG.md Sec 6.
cd /home/vexit/wavesplat/wavesplat || exit 1
LOG=tmp/heldout40_2026-07-26/chain_gpu.log
: > "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || { echo "ATOMS GATE FAIL" >> "$LOG"; exit 1; }
import json, sys
rows = [json.loads(l) for l in open('tmp/heldout40_2026-07-26/atoms_cache.jsonl')]
errs = sum(1 for r in rows if r['err'] is not None)
uniq = len({(r['episode'], r['diff_seed']) for r in rows})
print(f"atoms cache: rows={len(rows)} uniq={uniq} errs={errs}")
assert len(rows) == 200 and uniq == 200 and errs == 0, "ATOMS GATE FAIL"
print("ATOMS GATE: PASS")
PY

echo "=== STEP 2 learned arm ===" >> "$LOG"
nice -n 10 python3 tmp/heldout40_2026-07-26/run_arms_heldout40.py \
  --arms learned --workers 6 --out tmp/heldout40_2026-07-26/arm_runs.jsonl >> "$LOG" 2>&1
echo "LEARNED-EXIT $?" >> "$LOG"

echo "=== STEP 3 v2 integrity control ===" >> "$LOG"
nice -n 10 python3 tmp/heldout40_2026-07-26/run_arms_heldout40.py \
  --arms v2_konly --workers 6 --out tmp/heldout40_2026-07-26/arm_runs_v2_control.jsonl >> "$LOG" 2>&1
echo "V2CTRL-EXIT $?" >> "$LOG"

echo "=== STEP 5 auto live validation ===" >> "$LOG"
nice -n 10 python3 tmp/heldout40_2026-07-26/run_arms_heldout40.py \
  --arms auto --workers 6 --out tmp/heldout40_2026-07-26/arm_runs_auto_live.jsonl >> "$LOG" 2>&1
echo "AUTOLIVE-EXIT $?" >> "$LOG"
echo "CHAIN-GPU-DONE" >> "$LOG"
