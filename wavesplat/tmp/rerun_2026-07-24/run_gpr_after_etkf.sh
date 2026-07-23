#!/bin/bash
cd /home/vexit/wavesplat/wavesplat
D=tmp/rerun_2026-07-24
until [ "$(grep -c '"nRMSE"' $D/etkf_100ep_tuned_runs.jsonl 2>/dev/null)" -ge 100 ]; do sleep 20; done
echo "etkf done, GPR 100ep start $(date)"
python3 $D/gpr_100ep.py > $D/gpr_100ep.log 2>&1
echo "ALL BASELINES DONE $(date)"
