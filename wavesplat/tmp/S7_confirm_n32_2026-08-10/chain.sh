#!/usr/bin/env bash
# S7 실행 체인 — 생성 완료 대기 → 실행 전 게이트 → 배치.
# ⛔게이트가 FAIL 이면 배치를 돌리지 않는다 (PREREG §3.1). `&&` 가 그 규칙을 집행한다.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=tmp/S7_confirm_n32_2026-08-10

echo "[chain] $(date -Iseconds) 생성 완료 대기"
while true; do
  n=$(ls data_s7_crowded_2026-08-10/*.h5 data_s7_sparse_2026-08-10/*.h5 2>/dev/null | wc -l)
  if [ "$n" -ge 64 ] && ! pgrep -f 'gen_s7.py' >/dev/null; then break; fi
  sleep 60
done
echo "[chain] $(date -Iseconds) 에피소드 $n 개 — 게이트 실행"

python3 $D/gate.py 2>&1 | tee $D/gate.txt
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[chain] ⛔ GATE FAIL (rc=$rc) — S7 을 돌리지 않는다."
  exit 1
fi

echo "[chain] $(date -Iseconds) GATE PASS — 1152 런 시작"
WORKERS=${WORKERS:-8} python3 $D/run_s7.py 2>&1 | tee -a $D/run.log
echo "[chain] $(date -Iseconds) 배치 종료 rc=${PIPESTATUS[0]}"
