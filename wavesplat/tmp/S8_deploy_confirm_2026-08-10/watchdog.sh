#!/usr/bin/env bash
# S8 워치독 — 배치가 "돌고 있음"이 아니라 "제대로 돌고 있음"을 5분마다 확인한다.
# 정상 종료(배치 완주) 시 rc=0, **문제 감지 시 즉시 rc≠0 으로 빠져나온다**(그때 호출자가 깨어난다).
#
#   rc=0  배치 완주 (1920/1920)
#   rc=2  에러 레코드 발생
#   rc=3  진행 정체 (STALL_MIN 분 동안 레코드 증가 0)
#   rc=4  체인/러너 프로세스 사망 (미완주 상태)
#   rc=5  디스크 여유 부족
set -u
cd /home/vexit/wavesplat/wavesplat || exit 9
D=tmp/S8_deploy_confirm_2026-08-10
R=$D/results.jsonl
TOTAL=1920
POLL=${POLL:-300}
STALL_MIN=${STALL_MIN:-25}

last=-1
stall=0
while true; do
  n=$( [ -f "$R" ] && wc -l < "$R" || echo 0 )
  # grep -c 는 0건일 때 종료코드 1 이라 `|| echo 0` 를 쓰면 "0\n0" 이 된다 — 출력만 받는다
  errs=$(grep -c '"error"' "$R" 2>/dev/null); errs=${errs:-0}
  alive=$(pgrep -f "$D/chain.sh" | wc -l)
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  ts=$(date '+%H:%M:%S')

  echo "[$ts] $n/$TOTAL | 에러 $errs | 체인 $alive | 여유 ${free_gb}G"

  if [ "$errs" -gt 0 ]; then
    echo "WATCHDOG rc=2 — 에러 레코드 $errs 건"; exit 2; fi
  if [ "$free_gb" -lt 5 ]; then
    echo "WATCHDOG rc=5 — 디스크 여유 ${free_gb}G"; exit 5; fi
  if [ "$n" -ge "$TOTAL" ]; then
    echo "WATCHDOG rc=0 — 완주 $n/$TOTAL"; exit 0; fi
  if [ "$alive" -eq 0 ]; then
    sleep 20
    n2=$( [ -f "$R" ] && wc -l < "$R" || echo 0 )
    if [ "$n2" -ge "$TOTAL" ]; then echo "WATCHDOG rc=0 — 완주"; exit 0; fi
    echo "WATCHDOG rc=4 — 체인 사망, $n2/$TOTAL 에서 멈춤"; exit 4; fi

  if [ "$n" -eq "$last" ]; then
    stall=$((stall + POLL))
    if [ "$stall" -ge $((STALL_MIN * 60)) ]; then
      echo "WATCHDOG rc=3 — ${STALL_MIN}분 정체 ($n/$TOTAL)"; exit 3; fi
  else
    stall=0
  fi
  last=$n
  sleep "$POLL"
done
