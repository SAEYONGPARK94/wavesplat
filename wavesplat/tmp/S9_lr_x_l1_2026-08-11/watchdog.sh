#!/usr/bin/env bash
# S9 배치 워치독 — "돌고 있음"이 아니라 "제대로 돌고 있음"을 확인한다. 토큰 0.
#   rc=0  배치 완주 (2016/2016)
#   rc=2  에러 레코드 발생
#   rc=3  진행 정체 (STALL_MIN 분 동안 증가 0)
#   rc=4  체인 사망 (미완주)
#   rc=5  디스크 여유 부족
#   rc=6  ★제어 축 누출 (레코드에 alpha_* 키 또는 S2_ALPHA 설정 흔적)
set -u
cd "$(dirname "$0")" || exit 9
R=results.jsonl
TOTAL=2016
POLL=${POLL:-300}
STALL_MIN=${STALL_MIN:-25}

last=-1; stall=0
while true; do
  n=$( [ -f "$R" ] && wc -l < "$R" || echo 0 )
  errs=$(grep -c '"error"' "$R" 2>/dev/null); errs=${errs:-0}
  leak=$(grep -c '"alpha_episode"' "$R" 2>/dev/null); leak=${leak:-0}
  alive=$(pgrep -fc 'run_s9.py|chain.sh'); alive=${alive:-0}
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  ts=$(date '+%H:%M:%S')

  echo "[$ts] $n/$TOTAL | 에러 $errs | 제어누출 $leak | 체인 $alive | 여유 ${free_gb}G"

  [ "$errs" -gt 0 ] && { echo "WATCHDOG rc=2 — 에러 레코드 $errs 건"; exit 2; }
  [ "$leak" -gt 0 ] && { echo "WATCHDOG rc=6 — ★제어 축 누출 $leak 건 (G-7 위반)"; exit 6; }
  [ "$free_gb" -lt 5 ] && { echo "WATCHDOG rc=5 — 디스크 ${free_gb}G"; exit 5; }
  [ "$n" -ge "$TOTAL" ] && { echo "WATCHDOG rc=0 — 완주 $n/$TOTAL"; exit 0; }
  if [ "$alive" -eq 0 ]; then
    sleep 20
    n2=$( [ -f "$R" ] && wc -l < "$R" || echo 0 )
    [ "$n2" -ge "$TOTAL" ] && { echo "WATCHDOG rc=0 — 완주"; exit 0; }
    echo "WATCHDOG rc=4 — 체인 사망, $n2/$TOTAL 에서 멈춤"; exit 4
  fi
  if [ "$n" -eq "$last" ]; then
    stall=$((stall + POLL))
    [ "$stall" -ge $((STALL_MIN * 60)) ] && {
      echo "WATCHDOG rc=3 — ${STALL_MIN}분 정체 ($n/$TOTAL)"; exit 3; }
  else
    stall=0
  fi
  last=$n
  sleep "$POLL"
done
