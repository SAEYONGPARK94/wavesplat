#!/usr/bin/env bash
# 방벽 전체 실행 — `cd /home/vexit/wavesplat/wavesplat && bash tmp/_audit/run_all.sh`
#
# ★열한 검사기가 서로 다른 결함 부류를 본다. **하나만 돌리면 나머지는 침묵한다.**
#   1--5 = 배치·게이트·상수 / 6--11 = **논문 문안**(2026-08-28 등록).
#   2026-08-23 에 이 넷이 각각 독립적으로 결함을 찾았다(그리고 셋 다 자기 자신의 결함도 찾았다).
# ⛔**"경고 0건" 을 "검사했다" 로 읽지 마라** — 검사기가 조용히 건너뛴 이력이 하루에 세 번 있었다.
set -u
fail=0
run() { echo; echo "════════ $1"; shift; "$@"; local rc=$?; [ $rc -ne 0 ] && fail=1; return 0; }

run "1. 배치 산출물 (PREREG·게이트·판정·동결·순서)" python3 tmp/_audit/check_batch_artifacts.py
run "2. 판정 문턱 (우연과 구분되는가 · 도달 가능한가)" python3 tmp/_audit/check_thresholds.py
run "3. 손복사 기준 상수 (자료에서 재도출되는가)" python3 tmp/_audit/find_handcopied_refs.py
run "4. 논문 수치 대조 (모든 수가 기록에 대응하는가)" python3 tmp/_audit/check_thesis_numbers.py
# ★5번은 **사전등록 시점에** 돌려야 값이 가장 크다(외부 동결 파일은 그때 이미 존재한다).
#   M2 `G-13` = 참조 키가 B-26 동결 행에 아예 없어 **자료에 대고는 통과 불가능**한 게이트였다.
#   자가진단이 픽스처로 통과했기 때문에 배치가 끝날 때까지 드러나지 않았다.
run "5. 게이트 외부 참조 키 (공허한 FAIL/PASS)" python3 tmp/_audit/check_gate_refs.py

# ─────────────────────────────────────────────────────────────────────────────
# ★2026-08-28 신설 — **논문 문안 검사기 6종**을 상시화한다.
#   사유: 이 저장소 누적 1위 결함("같은 문장이 N곳인데 한 곳만 고친다")이 08-28 하루에
#   **세 번** 재발했다(영문 초록 T3 미수신 · 헤드라인 5곳 M2 공시 누락 · 척추 지목 되감김 2곳).
#   ★그날 오전에 검사기가 이미 있었는데도 재발했다 --- 사유는 "검사하지 않아서" 가 아니라
#   ⟹ **① 등록되지 않아 아무도 돌리지 않았고 ② 돌려도 전부 exit 0 이라 판정하지 않았다** 이다.
#   그래서 등록과 **동시에** 6·7·8 에만 exit code 를 붙였다(오탐이 0으로 확인된 것만).
# ⚠자문/실패 구분 (등록 전 실측):
#   6 canonical : 폐기된 축자 문면 4종만 실패 --- 오탐 0건.
#   7 bans      : **(B) 단서 동반 의무**만 실패(위반 0건). **(A) 금지 문구는 자문**이다 ---
#                 논문이 금지 자체를 축자로 인용하므로 정상 HIT 8종이 늘 켜져 있다.
#   8 numbers   : verdict JSON 에서 기대 문자열을 **계산**해 대조(52건, 불일치 0) --- 오탐 0건.
#   9·10·11     : **목록을 뽑을 뿐 판정하지 않는다** --- 사람이 읽는 자문. 늘 exit 0.
run "6. 논문 척추 축자 (같은 문장이 N곳 · 되감김)"   python3 tmp/STATUS_2026-08-28/audit_thesis_canonical.py
run "7. 논문 인용금지 · 단서 동반 의무"              python3 tmp/STATUS_2026-08-28/audit_thesis_bans.py
run "8. 논문 수치 단일출처 (verdict JSON 대조)"      python3 tmp/STATUS_2026-08-28/audit_thesis_numbers.py
run "9. 초록 대칭 (국문 ↔ 영문 원자) [자문]"         python3 tmp/STATUS_2026-08-28/audit_thesis_abstracts.py
run "10. 논문 자기모순 [자문]"                       python3 tmp/STATUS_2026-08-28/audit_thesis_contradictions.py
run "11. 주장 원자 다중 출현 [자문]"                 python3 tmp/STATUS_2026-08-28/audit_thesis_claims.py

echo
echo "════════ 요약"
[ $fail -eq 0 ] && echo "✅검사기 전부 통과 (9·10·11 은 자문이므로 출력을 눈으로 읽어라)" || echo "⚠경고 있음 — 위 출력을 읽어라 (오탐일 수 있다; 판정하지 않는 도구다)"
echo "⛔이 스크립트는 시험(pytest)을 대신하지 않는다: python3 -m pytest wavesim/tests/ wavesplat/tests/ -q"
exit $fail
