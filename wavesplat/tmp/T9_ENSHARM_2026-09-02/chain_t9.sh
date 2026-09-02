#!/usr/bin/env bash
# 생성이 끝나면 본 실행 → 프레임 계측 → 분석 을 이어 돈다.
# ⛔러너가 에피소드 부재를 스스로 검사하므로(rc=2) 불완전 코퍼스에서 돌지 않는다.
set -u
cd "$(dirname "$0")"
DATA=../../data_t9_sparse_2026-09-02
for i in $(seq 1 240); do
  n=$(ls $DATA/*.h5 2>/dev/null | wc -l)
  [ "$n" -ge 64 ] && break
  sleep 60
done
n=$(ls $DATA/*.h5 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] 코퍼스 $n/64 — 본 실행 시작"
[ "$n" -ge 64 ] || { echo "⛔코퍼스 미완 ($n/64) — 중단"; exit 3; }
python3 run_t9.py            || { echo "⛔본 실행 실패"; exit 4; }
echo "[$(date +%H:%M:%S)] 본 실행 끝 — 프레임 계측"
python3 run_t9.py --timing   || { echo "⛔계측 실패"; exit 5; }
echo "[$(date +%H:%M:%S)] 계측 끝 — 분석"
python3 analyze_t9.py
echo "[$(date +%H:%M:%S)] 체인 완료 rc=$?"
