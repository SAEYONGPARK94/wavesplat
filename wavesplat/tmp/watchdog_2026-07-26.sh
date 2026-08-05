#!/bin/bash
# WATCHDOG: exits when both RESULT.md exist OR a stall is detected.
A=/home/vexit/wavesplat/wavesplat/tmp/heldout40_2026-07-26
B=/home/vexit/wavesplat/wavesplat/tmp/encounter_rescore_2026-07-26
LOG=/home/vexit/wavesplat/wavesplat/tmp/watchdog_2026-07-26.log
NOWMAX=2700   # 45 min

newest_age() {  # dir -> seconds since newest regular file mtime
  f=$(find "$1" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f1)
  if [ -z "$f" ]; then echo 999999; else echo $(( $(date +%s) - ${f%.*} )); fi
}
newest_file() {
  find "$1" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

until false; do
  ts=$(date '+%F %T')
  ac=$(wc -l < "$A/atoms_cache.jsonl" 2>/dev/null || echo 0)
  ar=$(wc -l < "$A/arm_runs.jsonl" 2>/dev/null || echo 0)
  av=$(wc -l < "$A/arm_runs_v2_control.jsonl" 2>/dev/null || echo 0)
  wsn=$(wc -l < "$B/ws_enc.jsonl" 2>/dev/null || echo 0)
  cdn=$(wc -l < "$B/cond_enc.jsonl" 2>/dev/null || echo 0)
  pA=$(pgrep -fc 'precompute_atoms_heldout40|run_arms_heldout40' || true)
  pB=$(pgrep -fc 'run_ws_enc.py|run_cond_enc.py|encounter_rescore.*\.py' || true)
  ageA=$(newest_age "$A"); ageB=$(newest_age "$B")
  rA=0; [ -f "$A/RESULT.md" ] && rA=1
  rB=0; [ -f "$B/RESULT.md" ] && rB=1
  echo "$ts A[atoms=$ac arms=$ar v2=$av proc=$pA age=${ageA}s RESULT=$rA] B[ws=$wsn cond=$cdn proc=$pB age=${ageB}s RESULT=$rB]" >> "$LOG"

  if [ "$rA" = 1 ] && [ "$rB" = 1 ]; then
    echo "EXIT_REASON=BOTH_COMPLETE" >> "$LOG"; echo "EXIT_REASON=BOTH_COMPLETE"; break
  fi
  if [ "$rA" = 0 ] && [ "$ageA" -gt "$NOWMAX" ] && [ "${pA:-0}" -eq 0 ]; then
    echo "EXIT_REASON=STALL_A newest=$(newest_file $A) age=${ageA}s" >> "$LOG"
    echo "EXIT_REASON=STALL_A newest=$(newest_file $A) age=${ageA}s"; break
  fi
  if [ "$rB" = 0 ] && [ "$ageB" -gt "$NOWMAX" ] && [ "${pB:-0}" -eq 0 ]; then
    echo "EXIT_REASON=STALL_B newest=$(newest_file $B) age=${ageB}s" >> "$LOG"
    echo "EXIT_REASON=STALL_B newest=$(newest_file $B) age=${ageB}s"; break
  fi
  sleep 1200
done
