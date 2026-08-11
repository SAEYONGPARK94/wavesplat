#!/usr/bin/env bash
# S9 생성 워치독 — "돌고 있음"이 아니라 "제대로 돌고 있음"을 확인한다.
# 토큰 0 (bash 루프). 정상 완주 또는 문제 감지 시에만 종료 → 그때 호출자가 깨어난다.
#
#   rc=0  생성 완주 (96/96: sparse 64 + crowded 32)
#   rc=2  gen.log 에 ERROR 기록
#   rc=3  진행 정체 (STALL_MIN 분 동안 파일 증가 0)
#   rc=4  gen_s9.py 사망 (미완주 상태)
#   rc=5  디스크 여유 부족
#   rc=6  메시 판정이 ADEQUATE 가 아닌 에피소드 발생 (G-2 사전 게이트를 미리 거른다)
set -u
cd /home/vexit/wavesplat/wavesplat || exit 9
D=tmp/S9_lr_x_l1_2026-08-11
SP=data_s9_sparse_2026-08-11
CR=data_s9_crowded_2026-08-11
TOTAL=96
POLL=${POLL:-300}
STALL_MIN=${STALL_MIN:-20}

last=-1
stall=0
while true; do
  ns=$(ls "$SP"/*.h5 2>/dev/null | wc -l)
  nc=$(ls "$CR"/*.h5 2>/dev/null | wc -l)
  n=$((ns + nc))
  errs=$(grep -c 'ERROR' "$D/gen.log" 2>/dev/null); errs=${errs:-0}
  bad=$(grep -c 'mesh=' "$D/gen.log" 2>/dev/null); bad=${bad:-0}
  badmesh=$(grep 'mesh=' "$D/gen.log" 2>/dev/null | grep -vc 'mesh=ADEQUATE'); badmesh=${badmesh:-0}
  alive=$(pgrep -fc 'gen_s9.py'); alive=${alive:-0}
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  ts=$(date '+%H:%M:%S')

  echo "[$ts] $n/$TOTAL (sparse $ns/64, crowded $nc/32) | 에러 $errs | 비ADEQUATE $badmesh | gen $alive | 여유 ${free_gb}G"

  if [ "$errs" -gt 0 ]; then
    echo "WATCHDOG rc=2 — gen ERROR $errs 건"; exit 2; fi
  if [ "$badmesh" -gt 0 ]; then
    echo "WATCHDOG rc=6 — 비ADEQUATE 메시 $badmesh 건 (G-2 사전 탈락)"; exit 6; fi
  if [ "$free_gb" -lt 5 ]; then
    echo "WATCHDOG rc=5 — 디스크 여유 ${free_gb}G"; exit 5; fi
  if [ "$n" -ge "$TOTAL" ]; then
    echo "WATCHDOG rc=0 — 생성 완주 $n/$TOTAL"; exit 0; fi
  if [ "$alive" -eq 0 ]; then
    sleep 30
    ns=$(ls "$SP"/*.h5 2>/dev/null | wc -l); nc=$(ls "$CR"/*.h5 2>/dev/null | wc -l)
    n2=$((ns + nc))
    if [ "$n2" -ge "$TOTAL" ]; then echo "WATCHDOG rc=0 — 완주"; exit 0; fi
    echo "WATCHDOG rc=4 — gen_s9.py 사망, $n2/$TOTAL 에서 멈춤"; exit 4; fi

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
