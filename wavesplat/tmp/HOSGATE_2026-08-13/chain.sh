#!/usr/bin/env bash
# HOSGATE 전량 실행 체인. 어느 단계에서 죽어도 **같은 명령을 다시 치면 이어서** 간다
# (생성은 파일 존재로 skip, 추정은 results.jsonl 의 (episode,arm) 로 skip).
#
#   bash tmp/HOSGATE_2026-08-13/chain.sh
#
# 예상 5.3-5.8 h. ⚠생성과 추정을 **동시에 돌리지 말 것** — CPU 가 공유 병목이고
# "동시 실행으로 벌 수 있다" 는 이 저장소에서 이미 두 번 틀린 주장이다.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1
B=tmp/HOSGATE_2026-08-13

echo "=== [1/4] 코퍼스 생성 (32쌍, 기존 파일은 skip) $(date -Is)"
python3 $B/gen_hos.py all                 2>&1 | tee -a $B/chain.log || exit 1

echo "=== [2/4] 게이트 G-1..G-7 $(date -Is)"
python3 $B/run_hosgate.py gates           2>&1 | tee    $B/GATES.log || \
  { echo "GATES FAIL -- 배치 중단"; exit 1; }

echo "=== [3/4] 추정 3팔 x 64 에피소드 $(date -Is)"
python3 $B/run_hosgate.py run             2>&1 | tee -a $B/chain.log || exit 1

echo "=== [4/4] 분석 (PREREG 규약) $(date -Is)"
python3 $B/analyze_hosgate.py             2>&1 | tee    $B/ANALYSIS.log

echo "=== 완료 $(date -Is)"
