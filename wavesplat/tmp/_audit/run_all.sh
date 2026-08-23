#!/usr/bin/env bash
# 방벽 전체 실행 — `cd /home/vexit/wavesplat/wavesplat && bash tmp/_audit/run_all.sh`
#
# ★네 검사기가 서로 다른 결함 부류를 본다. **하나만 돌리면 나머지 셋은 침묵한다.**
#   2026-08-23 에 이 넷이 각각 독립적으로 결함을 찾았다(그리고 셋 다 자기 자신의 결함도 찾았다).
# ⛔**"경고 0건" 을 "검사했다" 로 읽지 마라** — 검사기가 조용히 건너뛴 이력이 하루에 세 번 있었다.
set -u
fail=0
run() { echo; echo "════════ $1"; shift; "$@"; local rc=$?; [ $rc -ne 0 ] && fail=1; return 0; }

run "1. 배치 산출물 (PREREG·게이트·판정·동결·순서)" python3 tmp/_audit/check_batch_artifacts.py
run "2. 판정 문턱 (우연과 구분되는가 · 도달 가능한가)" python3 tmp/_audit/check_thresholds.py
run "3. 손복사 기준 상수 (자료에서 재도출되는가)" python3 tmp/_audit/find_handcopied_refs.py
run "4. 논문 수치 대조 (모든 수가 기록에 대응하는가)" python3 tmp/_audit/check_thesis_numbers.py

echo
echo "════════ 요약"
[ $fail -eq 0 ] && echo "✅네 검사기 전부 통과" || echo "⚠경고 있음 — 위 출력을 읽어라 (오탐일 수 있다; 판정하지 않는 도구다)"
echo "⛔이 스크립트는 시험(pytest)을 대신하지 않는다: python3 -m pytest wavesim/tests/ wavesplat/tests/ -q"
exit $fail
