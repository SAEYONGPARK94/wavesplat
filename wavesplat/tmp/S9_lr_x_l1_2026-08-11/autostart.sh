#!/usr/bin/env bash
# S9 자동 진행 — 생성 완료를 기다렸다가 체인(게이트→런→채점)을 건다. 토큰 0.
#
# ★안전: `chain.sh` 는 **fail-closed** 다. 동결 md5 가 바뀌었거나 게이트가 FAIL 하면
#   런을 돌리지 않고 죽는다. 그러므로 무인 자동시작이 사전등록을 우회하지 못한다.
#
#   rc=0   배치 완주
#   rc=1   게이트 FAIL 또는 동결 위반 (⟹ 런이 시작조차 안 됨)
#   rc=7   생성이 STALL (에피소드가 GEN_STALL_MIN 분 동안 안 늘어남)
#   그 외  watchdog.sh 의 rc 를 그대로 전달 (2 에러 · 3 정체 · 4 사망 · 5 디스크 · 6 제어누출)
set -u
cd "$(dirname "$0")" || exit 9
SP=../../data_s9_sparse_2026-08-11
CR=../../data_s9_crowded_2026-08-11
NEED=96
GEN_STALL_MIN=${GEN_STALL_MIN:-25}

echo "[$(date '+%H:%M:%S')] 생성 완료 대기 (목표 $NEED)"
last=-1; stall=0
while true; do
  n=$(( $(ls $SP/*.h5 2>/dev/null | wc -l) + $(ls $CR/*.h5 2>/dev/null | wc -l) ))
  [ "$n" -ge "$NEED" ] && break
  if [ "$n" -eq "$last" ]; then
    stall=$((stall + 300))
    if [ "$stall" -ge $((GEN_STALL_MIN * 60)) ]; then
      echo "[$(date '+%H:%M:%S')] ⛔ 생성 정체 $n/$NEED — 중단"; exit 7; fi
  else
    stall=0
  fi
  last=$n
  sleep 300
done

echo "[$(date '+%H:%M:%S')] ✅ 생성 완료 $n/$NEED — 체인 시작"
nohup bash chain.sh > chain.log 2>&1 &
sleep 60                                  # 게이트가 판정을 낼 시간
if ! pgrep -f 'chain.sh' > /dev/null && [ ! -s results.jsonl ]; then
  echo "[$(date '+%H:%M:%S')] ⛔ 체인이 게이트에서 멈췄다 — gate.txt 확인"
  tail -20 gate.txt 2>/dev/null
  exit 1
fi
echo "[$(date '+%H:%M:%S')] 체인 가동 — 배치 워치독으로 전환"
exec bash watchdog.sh
