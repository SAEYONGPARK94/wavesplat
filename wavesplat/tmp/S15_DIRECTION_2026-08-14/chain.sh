#!/usr/bin/env bash
# S15_DIRECTION 전량 실행 — 한 줄로 끝까지 간다.
#
#     bash tmp/S15_DIRECTION_2026-08-14/chain.sh
#
# 각 단계는 **재개 가능**하고, 게이트가 실패하면 그 자리에서 멈춘다(사망조건을 미검정으로
# 남기지 않기 위해). 스레드 수가 단계마다 다른 이유는 SMOKE.md 참조:
#   wavesplat 팔 = 프로세스 병렬이 이긴다 (워커 6 x 스레드 1)
#   LETKF_PLUS   = BLAS 병렬이 이긴다     (워커 2 x 스레드 8)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
WS_ARMS='ORA-WPHI-0STEP@L3,ORA-WPHI-NT@L3,ORA-WPHI-0STEP@L4,ORA-WPHI-NT@L4,ORA-K-NT@L4,AUTO-NT@L4,AUTO-NT-DUP@L4,SHIP,ZERO'

step() { echo; echo "=== [$(date +%T)] $* ==="; }

step "1/6 dir180 32 에피소드 생성 (재개 가능)"
python3 "$HERE/gen_s15.py" 2>&1 | tee -a "$HERE/chain.log"

step "2/6 게이트 G-1..G-11"
python3 "$HERE/run_s15.py" gates 2>&1 | tee -a "$HERE/GATES.log"

step "3/6 ★G-0 S4 재현 (혼잡 16, seed 1557)"
S15_THREADS=1 python3 "$HERE/run_s15.py" g0 --workers 6 2>&1 | tee -a "$HERE/G0.log"

step "4/6 전량 — wavesplat 팔 (워커 6 x 스레드 1)"
S15_THREADS=1 python3 "$HERE/run_s15.py" run "$WS_ARMS" --workers 6 2>&1 | tee -a "$HERE/RUN_ws.log"

step "5/6 전량 — LETKF_PLUS (워커 2 x 스레드 8)"
S15_THREADS=8 python3 "$HERE/run_s15.py" run LETKF_PLUS --workers 2 2>&1 | tee -a "$HERE/RUN_letkf.log"

step "6/6 채점 (사전등록 그대로)"
python3 "$HERE/analyze_s15.py" 2>&1 | tee "$HERE/ANALYSIS.log"

echo
echo "완료. 판정은 ANALYSIS.log 마지막 줄. RESULT.md 는 사람이 쓴다."
