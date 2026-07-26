#!/usr/bin/env bash
# Wait for the held-out-40 generation to finish, VERIFY the adequacy gate on all 40
# from the manifest, then run the PURE-V2 arm.  Refuses to start the arm if any
# episode is not status=ok / mesh_verdict=ADEQUATE (PREREG Sec 3.3).
set -u
cd /home/vexit/wavesplat/wavesplat

DIR=data_heldout40_2026-07-26
LOG=tmp/heldout40_2026-07-26

# 1. wait for the driver to exit (it exits only after all jobs are consumed)
while kill -0 21756 2>/dev/null; do sleep 300; done
echo "GEN-DRIVER-EXITED $(ls $DIR/*.h5 2>/dev/null | wc -l)/40 episodes on disk"

# 2. adequacy gate, from the manifest, in code
python3 - <<'PY' > $LOG/adequacy_gate.txt 2>&1
import json, sys
m = json.load(open('data_heldout40_2026-07-26/_heldout40_manifest.json'))
bad = {k: v for k, v in m.items()
       if v.get('status') != 'ok' or v.get('mesh_verdict') != 'ADEQUATE'}
print(f"episodes in manifest: {len(m)}")
print(f"status=ok AND mesh_verdict=ADEQUATE: {len(m) - len(bad)}")
for k, v in bad.items():
    print(f"  NOT-OK {k}: status={v.get('status')} verdict={v.get('mesh_verdict')} "
          f"err={str(v.get('error'))[:120]}")
ok = (len(m) == 40 and not bad)
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
GATE=$?
cat $LOG/adequacy_gate.txt
if [ $GATE -ne 0 ]; then
  echo "ADEQUACY GATE FAILED -- v2 arm NOT started"
  exit 1
fi

# 3. pure-v2 arm, batchC protocol verbatim, <=6 workers
echo "ADEQUACY GATE PASS -- starting PURE-V2 arm"
WORKERS=6 nice -n 10 python3 tmp/heldout40_2026-07-26/v2_arm.py > $LOG/v2_arm.log 2>&1
echo "V2-ARM-EXIT $?"
tail -3 $LOG/v2_arm.log
