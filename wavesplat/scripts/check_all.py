#!/usr/bin/env python3
"""★방벽 전부를 한 명령으로 --- `python3 scripts/check_all.py`.

⛔이 스크립트가 존재하는 이유(2026-09-02):
방벽이 다섯 종인데 **전부 수동 실행**이었고 CI 도 훅도 0개였다. 그래서
`STATUS_2026-09-01/NEXT.md §2-④` 가 그것을 위생 항목으로 남겨 뒀다.
★이 저장소가 반복해 배운 것: **검사기는 사람이 기억해야 하면 안 돈다.**

포함:
  1. 논문 상수 생성기 `--check`      (손편집.드리프트 탐지)
  2. 동반 공시 검사기                 (병기 의무 11+7 규칙)
  3. 동반 공시 검사기 `--selftest`    (★항진 검사 --- 규칙이 규칙인가)
  4. 인용금지 양방향 래칫             (악화도 **개선도** 실패)
  5. `status_numbers.py`             (원장 문장 vs 동결 산출물, 런 0회)
  6. 배치 산출물 검사기               (verdict.짝.동결 md5)
  7. pytest                          (`--fast` 로 건너뛴다)

⚠**하나라도 실패하면 rc=1** 이다. 통과가 *"논문이 옳다"* 는 뜻은 **아니다** ---
막는 것은 **알려진 결함 부류**뿐이다.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = sys.executable

CHECKS = [
    # ★[2026-09-03] 적대적 감사 M1: `R1b` 가 `PREREG` 에 적은 `G-FREEZE`·`G-BLOCK` 을
    #   구현하지 않고도 `게이트 13/13` 을 인쇄했다 --- 사전등록 게이트 수와 **우연히 같아서**
    #   누락이 보이지 않았다. ⟹ 사전등록 표와 동결 verdict 를 래칫으로 대조한다.
    ('★게이트 구현 대조', [PY, 'tmp/_audit/check_gate_coverage.py']),
    # ★[2026-09-03] 적대적 심사가 찾았다: 동결 `verdict_b4.json` 이 **철회된 확증 라벨**을
    #   아직 싣고 있었고, 그 파일들이 `gen_paper_constants.py` 의 **입력**이다.
    ('★판정 파일 위생', [PY, 'tmp/_audit/check_verdict_hygiene.py']),
    # ★[2026-09-03 박사 기여 감사 C-10] `R5` 가 결과 당일 두 원고에 실리고 같은 날 철회됐다.
    #   ⟹ 원고가 인용하는 배치는 **검토 산출물**과 **등급 필드**를 가져야 한다.
    ('★원고 인용 성숙도', [PY, 'tmp/_audit/check_manuscript_maturity.py']),
    # ★★[2026-09-05] **MDE 상수 정합.** 수치 감사가 오늘 찾은 결함: 「검정력 .80」이라
    #   광고하는 MDE 여럿이 실제로는 0.79986~0.79995 였고(내림), **같은 양이 배치마다
    #   다르게 인쇄**됐다(`crit 19 @ n=28` = `T12` 0.7304 vs `T13` 0.7305).
    #   ⟹ 이 저장소의 이름 붙은 누적 결함 「손복사 기준 상수」가, 하필 ***모든 음성
    #   결과가 병기해야 하는 상수 안에서*** 재발했다. 계산의 유일출처를 만들고
    #   (`tmp/_audit/sign_test_mde.py`) 동결 verdict 를 그것으로 재계산해 대조한다.
    #   ⛔등록되지 않아 아무도 돌리지 않는 회귀 시험은 없는 것과 같다 ⟹ 자가진단 둘도 여기.
    ('★MDE 산수 자가진단', [PY, 'tmp/_audit/sign_test_mde.py', '--selftest']),
    ('★MDE 검사기 항진 검사', [PY, 'tmp/_audit/check_mde_constants.py', '--selftest']),
    # ★★[2026-09-05] **형제 배치 사이의 차단 게이트 소실.** `T16` 이 차단으로 신설한
    #   `G-INNER`(`gates_t16.py:553`, `PREREG` 5회)가 `T17` 에 **0회**다 — `gates_t17.py`.
    #   `analyze_t17.py`.`PREREG.md` 어디에도 없고 버렸다는 기록도 없다. 동결 행에 다시
    #   걸면 **2/72 셀**에서 깨지고 그 셀들은 전부 `edge_hit=False` ⟹ 남은 `G-EDGE` 는
    #   그 방향을 **구조적으로 볼 수 없다**(그 사실을 `T16 PREREG:505` 가 **미리** 적었다).
    #   ⛔등록되지 않아 아무도 돌리지 않는 회귀 시험은 없는 것과 같다 ⟹ 자가진단도 여기.
    ('★게이트 승계 항진 검사', [PY, 'tmp/_audit/check_gate_succession.py', '--selftest']),
    # ★★[2026-09-06] **끝점 선택이 실측 축의 검정력 벽이었다.** 원장의 상시 문장
    #   (*"n=12 는 10/12 를 어떤 자료에서도 기각 불가"*)은 **부호검정에 대해서만 참**이다 ---
    #   동결 `T17` 행에 정확 2^12 부호뒤집기 치환을 돌리면 **9/12 가 p=0.00195** 로
    #   Holm α/6=0.00833 을 통과한다(`tmp/_audit/ENDPOINT_SIGNFLIP.md`).
    #   ⟹ 계산의 유일출처를 만들었다(`tmp/_audit/signflip_test.py`, 형제 = `sign_test_mde.py`).
    #   ⛔**검정을 만들었으면 귀무에서 얼마나 틀리는지 먼저 재라** --- 그 도구의 자가진단은
    #   1종오류율·벡터화 동치·음성대조·`T17` 재현 핀을 함께 본다.
    #   ⛔등록되지 않아 아무도 돌리지 않는 회귀 시험은 없는 것과 같다.
    ('★치환검정 자가진단', [PY, 'tmp/_audit/signflip_test.py', '--selftest']),
    ('상수 생성기 --check', [PY, 'scripts/gen_paper_constants.py', '--check']),
    ('동반 공시', [PY, 'scripts/check_paper_disclosures.py']),
    ('★항진 검사', [PY, 'scripts/check_paper_disclosures.py', '--selftest']),
    # ★[2026-09-02] 감사 A6·A10·A11 이 셋 다 같은 원인을 지목했다 --- 위 검사기의 대상이
    #   `paper/` 하나라 **학위논문의 의무 공시는 아무도 기계로 보지 않았다**.
    # ★★[2026-09-03] 매크로에 걸린 공시 규칙을 **철자 표기**가 우회한다 --- 감사가
    #   `paper:2266` 의 *"eleven of twelve"* 로 증명했다(규칙 셋이 전부 침묵).
    ('★철자 비율', [PY, 'scripts/check_spelled_ratios.py']),
    ('★CLAUDE.md 시험 수', [PY, 'scripts/check_claudemd_counts.py']),
    ('학위논문 동반 공시', [PY, 'scripts/check_thesis_disclosures.py']),
    ('★학위 공시 항진 검사', [PY, 'scripts/check_thesis_disclosures.py', '--selftest']),
    # ★[2026-09-02] `N-1` 토론: 「제출 준비됨」의 단위는 원고가 아니라 **패키지**다.
    #   ⛔커버레터를 보는 기계가 0개였고, 실측하니 원고와 여섯 곳이 어긋나 있었다.
    ('★제출 패키지 항진 검사', [PY, 'scripts/check_submission_package.py', '--selftest']),
    ('인용금지 래칫', [PY, 'tmp/STATUS_2026-08-28/audit_paper_bans.py']),
    ('원장 수치 대조', [PY, 'tmp/STATUS_2026-08-30/status_numbers.py']),
    # ★[2026-09-02] 소거표 생성기. **조용히 빨간불이었다** --- `B-26` 게이트가 사후
    #   28/29 로 뒤집힌 뒤 아무도 `--check` 를 돌리지 않았고, 그 사이 남은 후보 목록이
    #   08-26 이후 손으로만 유지돼 낡았다(설계 감사가 그 사실로 시작했다).
    ('소거표 자기검증', [PY, 'tmp/STATUS_2026-08-26/elimination_table.py', '--check']),
    # ★[2026-09-02] 절대오차 표(런 0회 재집계). 생성기가 **교차 검증**을 한다:
    #   같은 팔이 두 배치에서 값이 다르면 죽는다. 재생성이 곧 검사다.
    ('절대오차 표 재생성', [PY, 'scripts/gen_abs_table.py']),
    # ★[2026-09-02] 프레임 예산 표 --- `T6` 동결 verdict 재인쇄. 게이트가 10/10 이 아니거나
    #   칸 집합이 바뀌면 죽는다. ★재생성이 곧 검사다.
    ('예산 표 재생성', [PY, 'scripts/gen_budget_table.py']),
    # ★[2026-09-02] `B-15` 그림 --- 동결 verdict + 동결 행 **재집계**이고, 둘이 어긋나면
    #   생성기가 죽는다(위조 시험으로 확인). ★재생성이 곧 검사다.
    ('raydrop 그림 재생성', [PY, 'scripts/gen_fig_raydrop.py']),
    # ★[2026-09-02] 추정 비교 그림 --- 근거리 칸을 `verdict_m10.primary.abs_median` 과,
    #   짝 승수를 `verdict_b12.ship_vs_spec` 과 대조하고 어긋나면 죽는다.
    ('추정비교 그림 재생성', [PY, 'scripts/gen_fig_estcompare.py']),
    ('실측 그림 재생성', [PY, 'scripts/gen_fig_real.py']),
    # ⛔[2026-09-02] **학위논문 상수 생성기가 방벽에 없었다** ⟹ 조용히 표류했다
    #   (오늘 테스트를 둘 추가하자 `\DeployNumTests` 가 261 로 낡았다).
    #   ★재생성이 곧 검사다 --- git 이 차이를 보여준다.
    ('학위 상수 재생성', [PY, 'scripts/gen_deploy_constants.py']),
    # ★[2026-09-05] `T11`~`T14` 상수. 학위논문에만 들어가므로 `paper_constants` 가
    #   아니라 별도 파일이다(두 원고의 개정 주기를 묶지 않는다). 동결 verdict 와
    #   어긋나면 `AssertionError` 로 죽는다 --- ★재생성이 곧 검사다.
    ('학위 T11--T14 상수 재생성', [PY, 'scripts/gen_thesis_t11t14_constants.py']),
    # ★[2026-09-02] 해상상태 표 생성기 --- ⛔학위논문이 **시뮬레이터에 없는 해상상태**를
    #   네 자리에 인쇄하고 있었다(BF3 0.88/5.0 · BF4 1.88/7.5 vs 코드 0.6/5.5 · 1.2/7.0).
    #   이 생성기는 `SEA_STATE` 를 임포트하고 **실제 에피소드 파일과 대조**한다
    #   ⟹ 코드나 코퍼스가 움직이면 죽는다. ★재생성이 곧 검사다.
    ('해상상태 표 재생성', [PY, 'scripts/gen_seastate_table.py']),
    # ★★[2026-09-02] **표준 사실 검사** --- 이 저장소의 누적 1위 결함(「정정이 한 곳에서
    #   멈춘다」)을 기계로 막는다. 오늘 하루에만 여섯 번 나왔고 셋은 **실험 조건 자체**였다
    #   (해상상태 네 표 · 채점 격자 2.5배 · 관측 수 네 자리).
    #   ⚠*"옳게 적혔는가"* 가 아니라 *"틀리게 적힌 것이 남았는가"* 를 본다 --- 후자만
    #   기계로 확실하다. 새 틀린 값은 생성기(코드에서 생성 + 코퍼스 대조)가 막는다.
    ('표준 사실 (코드 대 문서)', [PY, 'scripts/check_canonical_facts.py']),
    # ★[2026-09-02] 국문/영문 초록 원자 대조. ⛔감사가 *"KO/EN abstract split"* 을
    #   지적했고 실측하니 **국문만 `T-4` 를 보고**하고 영문에는 그 문단이 없었다.
    #   ★tex 주석이 그 위험을 이미 알고 *"영문 초록에 동시 반영한다"* 를 여러 곳에
    #   적어 뒀다 --- 사람의 기억에 맡기던 것을 기계로 옮긴다.
    ('초록 KO/EN 대조', [PY, 'scripts/check_abstract_mirror.py']),
    # ★★[2026-09-02] **인쇄면 검사** --- 검증 감사의 마지막 지적(축자):
    #   *"All 13 barriers are green while the paper carries a predicateless sentence,
    #    a duplicated heading, eight stuttering references … there is **no
    #    print-surface check**."* 맞다. 오늘 내 수리가 인쇄를 **다섯 번** 깼고
    #   그 다섯은 전부 grep 한 줄로 잡힌다.
    #   ⚠문법을 판정하지 않는다 --- **알려진 손상 패턴**만 본다.
    # ⚠11 s(두 PDF 를 `pdftotext` 한다). 훅에서는 **논문만** 본다 --- 학위논문은
    #   전체 실행이 잡는다. 사유: 훅이 느리면 우회당한다.
    # ★★[2026-09-02] 신설 --- 오늘 **일곱 번** 나온 결함: 주석 블록을 끼워 넣을 때
    #   `old` 가 줄 중간에서 끝나면 남은 본문이 주석 꼬리로 딸려 들어가 **인쇄면에서
    #   사라진다**. 학위논문 **초록 둘**에서 필수 공시가 그렇게 사라졌고 조판은 성공했다.
    #   ⛔한 건만 공시 검사기가 잡았다(요구 자구가 사라져서) --- 나머지 여섯은 무탐지였다.
    ('주석 삼킴', [PY, 'scripts/check_comment_swallow.py']),
    ('★주석 삼킴 자가시험', [PY, 'scripts/check_comment_swallow.py', '--selftest']),
    ('인쇄면 손상', [PY, 'scripts/check_print_surface.py'] +
     (['--paper-only'] if '--fast' in sys.argv else [])),
    # ⚠배치 검사기는 **상시 rc=1** 이다 --- 역사적 배치 넷이 동결 증거를 못 갖췄고
    #   그건 지금 고칠 수 있는 것이 아니다(런이 이미 끝났다).
    #   ⛔상시 빨간 검사는 사람이 **무시하기 시작한다** ⟹ 종료코드가 아니라
    #   **문제 배치의 이름 집합**을 래칫한다(수 아니라 이름 = 교체를 못 숨긴다).
]

# ⚠느린 것 --- `--fast` 에서 건너뛴다(훅은 `--fast` 를 쓴다: 느린 훅은 우회당한다).
SLOW = [
    # ★[2026-09-02] ⛔`thesis/build.sh` 는 미정의 참조를 **세어서 인쇄만** 하고 rc=0 을
    #   냈다 --- 오늘 내가 넣은 오타 라벨 둘이 조용히 통과했다. 이제 차단하고,
    #   그 차단이 **셸로 전파**되는지 위조로 확인했다(위조 rc=1 · 원복 rc=0).
    ('학위논문 조판', ['bash', 'thesis/build.sh']),
    # ⛔[2026-09-02] 위 줄의 종전 주석 *"rc=0 이 정상"* 은 **틀렸다** --- 내가 rc 를
    #   파이프 뒤 `$?` 로 쟀다(오늘 `build.sh` 에서 저지른 것과 **같은 실수**).
    #   실제 rc=1 이고 경고 34건인데 **대부분 역사적**이라 소급 수리가 불가능하다.
    #   ⟹ 실패로 두면 상시 빨간불이 되고, 빼면 아무도 안 본다
    #   ⟹ **경고 개수를 래칫**한다(아래 `check_audit_suite`).
    ('pytest', [PY, '-m', 'pytest', 'wavesim/tests/', 'wavesplat/tests/', '-q']),
]

# ★2026-09-02 실측. 이 넷은 **런이 이미 끝난** 배치라 소급 수리가 불가능하다.
#   ⛔새 이름이 들어오면 실패한다 --- 그게 이 래칫의 전부다.
#   ⛔이름이 **빠져도** 실패한다: 배치가 조용히 사라지거나 개명된 것이므로 확인해야 한다.
FROZEN_EVIDENCE_KNOWN = {
    'B2_REAL_2026-08-23',
    'M1_NONLIN_REAL_2026-08-28',
    'M4_CAPACITY_2026-08-28',
    'VOID_unread',
}


# ★[2026-09-02] **`FROZEN.sha256` 사각지대.**
#   `tmp/_audit/check_batch_artifacts.py` 는 `FROZEN_md5*.txt` 만 본다 ⟹ 초기 배치들이
#   쓰던 `FROZEN.sha256` 은 **아무도 검증하지 않았다**. 실측: 셋 중 **둘이 실패**하고
#   그중 하나가 **배포 체크포인트를 만든 배치**다(`mixed_retrain_2026-07-26`).
#   ⚠둘 다 **소급 수리가 불가능**하다(동결이 커밋 시점에 이미 낡아 있었다 ---
#   상세 = `tmp/_audit/H2_CHECKPOINT_PROVENANCE_2026-09-02.md`).
#   ⟹ 배치 검사기와 같은 규약: **이름 집합 래칫**(새 이름이 들어오거나 기지 이름이
#   사라지면 실패). ⛔침묵시키지 않는다.
SHA256_KNOWN_FAIL = {'confirm40_2026-07-27', 'mixed_retrain_2026-07-26'}


AUDIT_WARN_BASELINE = 39   # 늘어나면 실패, 줄어도 실패(기록 강제).
# ★★[2026-09-05] 38 -> **39**. ⛔침묵이 아니다 --- 두 사실을 **따로** 적는다.
#   (a) ⛔**기준선 38 은 이미 낡아 있었다.** 초록 재작성을 시작하기 전 `HEAD` 의
#       `thesis_main.tex`.`t11t14_constants.tex`.생성기.감사기를 **되돌려 놓고**
#       `run_all.sh` 를 돌려 **42**를 실측했다(파일 스왑 뒤 원복, 런 0회).
#       ⟹ +4 는 **이 작업이 만든 것이 아니고**, 그 출처를 이 작업에서 확정하지 못했다.
#       ★기준선 자신이 「손복사 기준 상수」 결함을 또 낸 것이다(아래 2026-09-02 교훈 참조).
#   (b) ★**이 작업은 경고를 42 -> 39 로 줄였다.** 초록 재작성이 KO/EN **갈림 경고 셋**을
#       없앴다: 「사다리 고리 수(여섯 vs six)」·「실측 레코드 수(12 vs real)」·
#       「제4장 제어 지위(기여로 주장하지 않 vs does not claim)」 --- 두 초록이 같은
#       것을 같은 방식으로 말하게 됐기 때문이다.
#   ⚠그래서 39 는 **개선의 기록**이지 새 결함의 수용이 아니다. 재측정 절차 = 위 (a).
# ★[2026-09-05] 36 -> **38** 로 **의도적** 상향. 두 단위이고 사유가 다르다 ---
#   ⛔둘을 하나로 뭉뚱그리지 않는다.
#   (a) **+1 은 이 작업 이전에 이미 있었다.** `T11`~`T14` 학위논문 반영을 시작하기 전,
#       아무것도 고치지 않은 상태에서 `run_all.sh` 를 돌려 **37**을 실측했다(기준선 36).
#       ⚠그 시점의 작업트리는 `paper/` 가 다른 작업 단위 때문에 더러웠으므로
#       **그 +1 의 출처를 이 작업에서 확정하지 못했다** --- 침묵시키지 않고 여기 적는다.
#   (b) **+1 은 이 작업의 것이고 결함이 아니다.** `thesis/t11t14_constants.tex` 를
#       신설해 `ls paper/*.tex thesis/*.tex` 가 10 -> **11** 이 되었고,
#       `check_thesis_numbers.py` 는 **문서마다** 「(C) 값 일치 상한」 줄을 하나 낸다.
#       ⟹ 34 -> 35 상향 때와 **같은 사유**(문서 수 증가)다. 확인 = 위 `ls` 가 11.
# ⛔[2026-09-02 저녁] 종전 값 **35 는 한 번도 참이 아니었다.** 주석이 *"실측"* 이라
#   적었지만, 그 값을 도입한 커밋 `b7b07d62` 에서 같은 표현으로 다시 세면 36 이고
#   `041ea0e0`·`HEAD~1`·현재도 36 이다 --- ★경고 **집합이 동일**하고 학위논문
#   줄번호만 밀렸다(diff 로 확인). ⟹ 오늘의 편집이 경고를 늘린 것이 아니다.
# ★교훈: 이 저장소가 사냥꾼까지 붙여 둔 **「손복사 기준 상수」** 결함이,
#   하필 **래칫 자신의 기준선**에서 재발했다. 기준선을 적을 때는 그 자리에서
#   `check_audit_suite()` 를 돌려 값을 받아 적어라(주석에 '실측' 이라 쓰는 것은 증거가 아니다).
#   34 -> 35 의 **의도적** 상향(2026-09-02, 6차 감사 수리):
#   `check_thesis_numbers.py:171` 이 `paper/*.tex` 를 전부 훑는데 오늘
#   `paper/budget_table.tex` 를 **새로 만들었다** ⟹ 그 문서 하나가 「(C) 값 일치
#   상한」 줄을 하나 더 낸다. ★결함이 아니라 **문서 수 증가**다. 확인 방법 =
#   `ls paper/*.tex thesis/*.tex | wc -l` 이 9 -> 10.
#   ⛔침묵시키지 않았다 --- 기준선을 올리고 사유를 여기 적는 것이 규약이다.


def check_audit_suite():
    """`tmp/_audit/run_all.sh` --- **판정 도구가 아니라 경고 도구**다.

    ⛔학위논문 감사 축자: *"`run_all.sh`, which **today nothing runs**"*.
    실제로 그 §3 이 **학위논문 그림 안의 금지 수치**를 세고 있었는데 아무도 안 봤다.
    ⚠rc=1 이 정상 상태(경고 34건, 대부분 역사적) ⟹ 실패로 두면 상시 빨간불이 되고
    빼면 다시 아무도 안 본다. ⟹ **개수를 래칫**한다: 늘어도 줄어도 실패한다.
    """
    r = subprocess.run(['bash', 'tmp/_audit/run_all.sh'], cwd=ROOT,
                       capture_output=True, text=True)
    n = sum(1 for l in (r.stdout + r.stderr).split('\n')
            if l.lstrip().startswith('⚠') or 'WARN' in l)
    if n != AUDIT_WARN_BASELINE:
        print(f'  [FAIL] 감사 도구 묶음 (래칫)  경고 {n} != 기준 {AUDIT_WARN_BASELINE}')
        print('         ⚠늘었으면 새 경고를 읽어라. 줄었으면 **의도적으로** 기준을 내려라.')
        return False
    print(f'  [OK  ] 감사 도구 묶음 (래칫: 경고 {n}건 그대로)')
    return True


# ★[2026-09-05] **시드 대역 충돌** — `tmp/_audit/check_seed_bands.py`.
#   계기: `T16` 설계가 `T15` 와 **완전히 같은 밑변**을 골랐다. 설계 에이전트는 형제 시드를
#   **동결 결과 파일**에서만 셌고 `T15` 는 같은 시각 동결 중이라 결과가 **아직 없었다**
#   ⟹ 구조적으로 안 보였다. ★부류: *"형제를 결과로만 세면 동시에 만들어지는 형제는 안 보인다."*
#   ⟹ 검사기가 **선언(소스 상수) + 관측(결과 행)** 을 둘 다 본다. `T16` 은 20261300 으로 옮겼다.
#
#   ⚠아래 넷은 **런이 끝난 배치**라 소급 수리가 불가능하다. 실측으로 확인한 범위:
#     · 판정에 쓰이는 **모든 층 안에서 시드는 블록과 1:1** 이다(층별 감사 확인)
#       ⟹ 부호검정의 블록 독립성은 이 사유로 깨지지 않는다.
#     · 어떤 verdict 도 시드를 공유하는 **두 가족을 합치지 않는다**
#       (`T12 SECR` 은 EXPLORATORY 이고 PRIMARY 와 별도 검정이다).
#   ⛔그러므로 **"판정이 무효" 가 아니다.**
#   ⚠★그리고 *"거짓 서술을 고쳐야 한다"* 도 **아니다** — 전수 확인 결과
#   `T11`~`T14` 의 어느 문서도 **배치 간 시드 대역 서로소를 주장하지 않는다**
#   (`T13/PREREG.md:97` 의 서로소 주장은 `FRESH-28` 대 `T11`+스모크에 한정되고 **참이다**;
#   `T11/NOTE.md:118,138` 의 서로소는 **파일** 서로소이지 시드가 아니다).
#   ⟹ 정확한 진술은 **"대역 서로소 규율이 어디에도 명시돼 있지 않았고 그래서 집행되지
#   않았다"** 이다. 이제 기계가 집행한다. ⛔침묵시키지 않는다.
#   ★[같은 날 오후] 독립 검증 에이전트가 이 검사기를 **모의 `tmp/` 위에서 공격해** 구멍
#   셋을 실증했다 ⟹ 전부 고쳤고 기지 집합이 4 → 5 로 늘었다:
#     A1 밑변이 **다르면서** 대역이 겹치면 통과했다(모의: FAIL 0) ⟹ 선언 대역 구간 교집합 검사
#     A2 폭 추출이 이 저장소의 **두 실제 표기 모두** 실패(튜플 대입 · 백틱) ⟹ 폭이 조용히 1 로
#        축소됐고 **「다음 안전한 밑변」 추천이 틀린 값**(T15 대역 안)을 냈다
#        ⟹ 두 표기를 파싱하고, 폭 미상은 **FAIL** 이며 추천을 **보류**한다
#        (⛔`(x or 기본값)` 부류 — 조용히 기본값으로 줄이지 않는다)
#     A3 스캔 범위가 `gates_*.py` 를 빠뜨려 거기 밑변을 둔 배치가 목록에서 통째로 누락
#   ⚠그리고 고치는 과정에서 **내 계수기가 자기 결함을 냈다** — 남은 `fails = 0` 이
#     폭-미상 FAIL 을 지워 요약이 5건을 4건으로 인쇄했다.
SEED_BAND_KNOWN = {
    'CROSS:T11_EXTACC_2026-09-03|T14_AUTOSCALE_2026-09-04',
    'CROSS:T12_ISOLATE_2026-09-04|T14_AUTOSCALE_2026-09-04',
    'CROSS:T13_DEPTHFIX_2026-09-04|T14_AUTOSCALE_2026-09-04',
    'INTRA:T12_ISOLATE_2026-09-04:PRIMARY|SECR',
    # ⚠`realspread_2026-08-05` 는 밑변만 선언하고 `N_…` 폭 상수가 없다. **위반이 확인된
    #   것이 아니라 판정 불가**다 — 그래서 이름이 `WIDTH:` 다. ⛔"괜찮다" 로 읽지 마라.
    'WIDTH:realspread_2026-08-05',
}


def check_seed_bands():
    """시드 대역 이름집합 래칫 — 새 충돌이 들어오거나 기지 충돌이 사라지면 실패."""
    r = subprocess.run([PY, 'tmp/_audit/check_seed_bands.py', '--keys'],
                       cwd=ROOT, capture_output=True, text=True)
    line = next((l for l in r.stdout.split('\n') if l.startswith('KEYS')), 'KEYS')
    got = set(line.split()[1:])
    if got != SEED_BAND_KNOWN:
        print('  [FAIL] 시드 대역 (래칫)')
        for k in sorted(got - SEED_BAND_KNOWN):
            print(f'         ⛔새 충돌: {k}  — 밑변을 옮겨라(검사기가 안전한 밑변을 인쇄한다)')
        for k in sorted(SEED_BAND_KNOWN - got):
            print(f'         ⚠기지 충돌이 사라졌다: {k}  — 배치가 개명/삭제됐는지 확인')
        return False
    print(f'  [OK  ] 시드 대역 (래칫: 기지 {len(got)}건 그대로)')
    return True


# ★★[2026-09-05] **MDE 상수 이름집합 래칫** — `tmp/_audit/check_mde_constants.py`.
#   규약(`tmp/_audit/sign_test_mde.py`)은 하나다: ***MDE 를 참값보다 낮게 보고하지 않는다***
#   (효과크기 `|q-0.5|` 를 0.5 에서 멀어지는 쪽으로 반올림 ⟹ 반환값 자신이 목표 검정력을
#   달성한다). 아래는 그 규약을 **어긴 채 동결된** 항목이고, ⛔**고치지 않는다 — 증거다.**
#
#   ★파급은 **없다**: MDE 는 라벨을 만들지 않는다(문턱 `w>=k` 가 만든다). 전수 확인 결과
#   `T11`~`T16` 의 어느 게이트도 MDE 를 수치 비교에 쓰지 않는다 ⟹ **라벨 이동 0건**.
#   부족분은 `T7b` 하나를 빼면 전부 <=5.0e-03 이고 대부분 <=2.2e-04 다.
#
#   ⛔★**`T7b` 는 반올림이 아니다.** `analyze_t7b.py:38` 이 `'q80': 0.7109` 를 **리터럴**로
#     적었는데(같은 줄의 나머지 넷은 전부 `G.` 에서 계산된다) 그 값은 **`T7`(n=48)의 것**이고
#     `T7b` 는 n=32·k=23 이라 참값이 **0.7653** 이다 ⟹ 인쇄값의 실제 검정력 **0.5495**.
#     ✅다행히 그 수는 `RESULT.md`·두 원고 어디에도 전파되지 않았다(grep 0건).
#
#   ⚠`UNREGISTERED:` 는 *"이 검사기의 산수로 재도출할 수 없는 양"* 이다(검사기가 사유를
#     인쇄한다). ⛔"괜찮다" 가 아니라 **판정 불가**다.
#   ★새 배치는 `mde` 블록에 `n`·`thresh`·`power` 를 실어라 — 그러면 자기기술로 자동 검사되고
#     등록이 필요 없다. 실으면 이 목록에 이름이 들어갈 일도 없다.
MDE_KNOWN = {
    # ── 내림으로 검정력이 문턱 바로 아래로 떨어진 것 (부족분 <=2.2e-04) ──────
    'T12_ISOLATE_2026-09-04:mde.sec_q80',            # 0.8332 → 검정력 0.79995
    'T12_ISOLATE_2026-09-04:mde.secr_q80',           # 0.7304 → 0.79990  (T13 은 0.7305)
    'T14_AUTOSCALE_2026-09-04:mde.primary_q80',      # 0.7630 → 0.79986
    'T14_AUTOSCALE_2026-09-04:mde.sec_q80',          # 0.7304 → 0.79990  (같은 양, 같은 드리프트)
    'T15_HOMEREG_2026-09-05:mde.cell_q80',           # 해침 다리 0.0975 → 0.79985
    'T15_HOMEREG_2026-09-05:mde.q80',                # 해침 다리 0.3185 → 0.79978
    'T16_BANDFIX_2026-09-05:mde.BLIND_q80',          # 0.7597 → 0.79994
    'T16_BANDFIX_2026-09-05:mde.D_q80',              # 0.7074 → 0.79991
    'T16_BANDFIX_2026-09-05:mde.F_q50',              # 0.6077 → 0.49981
    'T16_BANDFIX_2026-09-05:mde.SECE_q50',           # 0.6077 → 0.49981 (F 와 같은 설계)
    'T16_BANDFIX_2026-09-05:mde.S_q80',              # 0.6915 → 0.79984
    # ⚠`T16` 은 같은 양을 **두 곳에** 인쇄한다 — 둘 다 이름을 적는다(한쪽만 고치는 부류 방지).
    'T16_BANDFIX_2026-09-05:primary_family.D (착지 축, 양측, Bonferroni α/2).q80',
    'T16_BANDFIX_2026-09-05:SEC.SEC-BLIND (미접촉 24 · ⛔angle 과 완전 교락).q80',
    # `n_eff_table` — 동점 제외 대비표. `round()` 가 7개 칸에서 내림했다.
    'T16_BANDFIX_2026-09-05:n_eff_table.7.q80',
    'T16_BANDFIX_2026-09-05:n_eff_table.9.q80',
    'T16_BANDFIX_2026-09-05:n_eff_table.10.q80',
    'T16_BANDFIX_2026-09-05:n_eff_table.12.q80',
    'T16_BANDFIX_2026-09-05:n_eff_table.15.q80',
    'T16_BANDFIX_2026-09-05:n_eff_table.50.q80',
    'T16_BANDFIX_2026-09-05:n_eff_table.60.q80',
    # ── ⛔부류가 다른 것 (반올림이 아니라 **다른 배치의 상수**) ──────────────
    'T15_HOMEREG_2026-09-05:mde.q90',                # 0.2952 → 0.89499 (부족 5.0e-03)
    'T7b_SPECTRAL_2026-09-02:thresholds.q80',        # 0.7109 → 0.54946 (부족 2.5e-01)
    # ── ⚠판정 불가 (부호검정의 승률 MDE 가 아니다) ──────────────────────────
    'UNREGISTERED:M2_LOCTUNE_2026-08-28',
    'UNREGISTERED:R7_FLOOR_2026-09-03',
    # ── T18 (2026-09-05 추가). ⛔`T18` 은 `round(mde_q(...), 4)` 로 **가까운 쪽**
    #   반올림해서 규약(0.5 에서 **멀어지는** 쪽)보다 최대 **2.01e-04** 낮다.
    #   ★라벨은 정수 문턱이 만들고 MDE 는 `disclosures`/`mde` 블록에만 들어가므로
    #   **판정 불변**이다. 동결 배치라 고치지 않고 여기 등록한다(= 새 배치에만 규약 적용).
    'T18_FACTORIAL_2026-09-05:SEC.SEC-A (사전-OFF 열의 양자화 효과, 무보정 단측).q80',
    'T18_FACTORIAL_2026-09-05:SEC.SEC-B (양자화 열의 사전 효과, 무보정 단측).q80',
    'T18_FACTORIAL_2026-09-05:SEC.SEC-C (신선 시드 45·46·47 · 앞 16, 무보정).q80',
    'T18_FACTORIAL_2026-09-05:mde.I_q90',
    'T18_FACTORIAL_2026-09-05:mde.SEC_q80',
    'T18_FACTORIAL_2026-09-05:mde.SEC_q90',
    'T18_FACTORIAL_2026-09-05:mde.SECC_q80',
    'T18_FACTORIAL_2026-09-05:n_eff_table.16.q80',
    'T18_FACTORIAL_2026-09-05:n_eff_table.20.q80',
    'T18_FACTORIAL_2026-09-05:n_eff_table.28.q80',
    'T18_FACTORIAL_2026-09-05:n_eff_table.44.q80',
}


def check_mde_constants():
    """MDE 상수 이름집합 래칫 — 새 미달이 들어오거나 기지 미달이 사라지면 실패."""
    r = subprocess.run([PY, 'tmp/_audit/check_mde_constants.py', '--keys'],
                       cwd=ROOT, capture_output=True, text=True)
    got = {l.split('\t', 1)[1] for l in r.stdout.split('\n') if l.startswith('KEY\t')}
    if r.returncode != 0 and not got:
        print('  [FAIL] MDE 상수 (래칫)')
        print('         ⛔검사기가 키를 내지 못했다 — 죽었거나 출력 형식이 바뀌었다.')
        for line in (r.stdout + r.stderr).strip().split('\n')[-6:]:
            print(f'         {line}')
        return False
    if got != MDE_KNOWN:
        print('  [FAIL] MDE 상수 (래칫)')
        for k in sorted(got - MDE_KNOWN):
            print(f'         ⛔새 미달: {k}  — `sign_test_mde.mde_q()` 로 계산하라'
                  ' (⛔동결 배치는 고치지 말고 여기 등록 + 사유를 적어라)')
        for k in sorted(MDE_KNOWN - got):
            print(f'         ⚠기지 미달이 사라졌다: {k}  — 배치가 개명/삭제됐는지 확인')
        return False
    print(f'  [OK  ] MDE 상수 (래칫: 기지 {len(got)}건 그대로)')
    return True


# ★★[2026-09-05] **형제 승계 이름집합 래칫** — `tmp/_audit/check_gate_succession.py`.
#   규약: 배치 `S` 가 배치 `P` 의 모듈을 **실제로 임포트**하면(= 같은 규칙 계열·같은 측정면),
#   `P` 의 **차단** 게이트 이름 하나하나가 `S` 의 산출물(`.py`.`.md`.`verdict*.json`)에
#   **한 번은 나타나야 한다**. 승계하든, 이름을 바꾸든, *"…이므로 버린다"* 라고 적든 좋다.
#   ⛔막으려는 것은 **약한 승계**가 아니라 ***침묵***이다.
#
#   ⚠**아래는 이미 동결된 배치들이다 — 고치지 않는다.** 그리고 ⛔**"괜찮다" 로 읽지 마라**:
#   내가 실제로 재적용해 확인한 것은 **`T17 ← T16` 의 세 건뿐**이고 나머지 65건은
#   **미조사**다(열람 범위 = 주장 범위). 새 이름이 들어오면 실패하고, 기지 이름이
#   사라져도 실패한다(개명·삭제·수리를 구별하기 위해).
GATE_SUCCESSION_KNOWN = {
    # ── ★★이 검사기가 만들어진 사건: `T17` 이 `scale_rules_t16` 을 실행하면서 `T16` 의
    #    차단 게이트 37개 중 20개를 **한 번도 언급하지 않았다**. `T17` 은 라벨 없이 닫혔다.
    #    ⛔셋만 재적용해 확인했다(런 0회, 동결 `results_t17.jsonl` 432행 재집계):
    #      · `G-INNER`  = ⛔**실제로 깨진다** — `k̂₂ <= 3.5·k̂₁` 가 **충실한 재적용**
    #                     (`T16` 과 같은 양 = 셀 좌표 규칙 프로브)에서 **2/72**
    #                     (`BS13b|w2` 3.696 · `BS13c|w5` 3.632), `T17` 에만 있는 재라벨
    #                     프로브(`_prime`)에서 **3/72**(`BS13a|w5` 4.701 추가).
    #                     ★세 셀 **전부 `edge_hit = False`** 다.
    #      · `G-FLOOR`  = ✅재적용 시 **0/72**(최소 floor_margin 48.2). 잰 뒤에 안 것이다.
    #      · `G-KLIM1`  = ✅재적용 시 **0/72**(klim2 == 3.5·k̂₁ 전건 성립).
    #    ⚠`G-EQUIVAR` 는 **버려진 것이 아니라 개명**됐을 가능성이 높다(`T17 G-DEPTH-EXACT` 가
    #      같은 등변성 조문이다 — `tmp/_audit/PREREG_CHECKLIST.md` `B12`) ⟹ ★검사기는 개명을
    #      볼 수 없다. **그래서 의무가 "옛 이름을 한 번 적어라" 인 것이다.**
    #    ⚠`G-TORIGIN` 은 이미 독립적으로 발견돼 있었다(`PREREG_CHECKLIST` `B13`:
    #      *"`T16` 의 `G-TORIGIN` 은 `T17` 을 못 잡았을 것이다"*) — 그 절도 **소실 자체는
    #      적지 않았다**.
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-CAP',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-DENSITY',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-EQUIVAR',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-EQUIVAR-RAW',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-FINITE',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-FLOOR',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-GRID',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-INNER',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-KLIM1',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-KLIM1-LIVE',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-KTRAIN',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-LTRAIN',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-NOLEAK',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-POP',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-SAMPLE',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-SHIPID',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-SIZE',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-SRC-PIN',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-TORIGIN',
    'DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-WRAPPER',
    # ── `T17 ← T14`(`scale_rules_t14`). ⚠위 묶음의 **부분집합**이다 — `T16` 이 `T14` 를
    #    승계했으므로 같은 이름이 두 간선에 걸린다. ⛔합쳐 세지 마라(간선이 사실 단위다).
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-CAP',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-DENSITY',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-EQUIVAR',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-FINITE',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-KTRAIN',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-LTRAIN',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-NOLEAK',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-POP',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-SAMPLE',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-SHIPID',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-SIZE',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-SRC-PIN',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-TORIGIN',
    'DROP:T17_REALSCALE_2026-09-05|T14_AUTOSCALE_2026-09-04|G-WRAPPER',
    # ── `T15 ← T12`(`consts_t12` = 처치값 단일출처, `T15 analyze:123 G-TREATPIN`).
    #    ⚠간선은 **사실**이지만(그 모듈을 실행한다) 측정 질문은 `T12` 보다 좁다.
    #    ⛔11건 전부 **미조사**다 — *"해당 없음"* 이라 주장하지 않는다.
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-DENSITY',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-EFFECT',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-FREEZE',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-GROWFIRE',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-HELDOUT',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-RESCALE',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-RNGSTREAM',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-SAMPLE',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-SHIPPIN',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-SIZE',
    'DROP:T15_HOMEREG_2026-09-05|T12_ISOLATE_2026-09-04|G-TORIGIN',
    # ── `T` 계열의 나머지(`gates_t11`/`run_t11` 재사용). ⚠미조사.
    'DROP:T12_ISOLATE_2026-09-04|T11_EXTACC_2026-09-03|G-POP',
    'DROP:T12_ISOLATE_2026-09-04|T11_EXTACC_2026-09-03|G-WRAPPER',
    'DROP:T13_DEPTHFIX_2026-09-04|T11_EXTACC_2026-09-03|G-SIZE',
    'DROP:T16_BANDFIX_2026-09-05|T11_EXTACC_2026-09-03|G-RESCALE',
    # ── `C1`(모델갭) 계열 — `gates_c1` 을 임포트하는 다섯 배치가 `G-COLS` 를 공통으로
    #    떨어뜨렸다. ⚠미조사. ★이 다섯이 **같은 이름**을 떨어뜨린 것은 우연이 아닐 수
    #    있다(공유 조상에서 한 번 빠지면 자손 전체로 전파된다) — 그것 자체가 소견이다.
    'DROP:C1B_BESTFIT_2026-09-03|C1_MODELGAP_2026-09-03|G-COLS',
    'DROP:C1C_BESTFIT_2026-09-03|C1_MODELGAP_2026-09-03|G-COLS',
    'DROP:C2_STOKES_2026-09-03|C1_MODELGAP_2026-09-03|G-COLS',
    'DROP:E2_EXTSYNTH_2026-09-03|C1_MODELGAP_2026-09-03|G-COLS',
    'DROP:E2_EXTSYNTH_2026-09-03|C1_MODELGAP_2026-09-03|G-CTRL',
    'DROP:E2_EXTSYNTH_2026-09-03|C1_MODELGAP_2026-09-03|G-NORUN',
    'DROP:T10_EXTDISP_2026-09-03|C1_MODELGAP_2026-09-03|G-0b',
    'DROP:T10_EXTDISP_2026-09-03|C1_MODELGAP_2026-09-03|G-COLS',
    'DROP:T10_EXTDISP_2026-09-03|C1_MODELGAP_2026-09-03|G-CTRL',
    'DROP:T1B_THIRDPARTY_2026-09-03|C1_MODELGAP_2026-09-03|G-0b',
    'DROP:T1B_THIRDPARTY_2026-09-03|C1_MODELGAP_2026-09-03|G-COLS',
    'DROP:T1B_THIRDPARTY_2026-09-03|C1_MODELGAP_2026-09-03|G-CTRL',
    'DROP:T1_THIRDPARTY_2026-09-03|C1_MODELGAP_2026-09-03|G-0b',
    'DROP:T1_THIRDPARTY_2026-09-03|C1_MODELGAP_2026-09-03|G-COLS',
    'DROP:T1_THIRDPARTY_2026-09-03|C1_MODELGAP_2026-09-03|G-CTRL',
    # ── `E1` → `E2`(외부 기준선 계열, `extarm_e1`.`gates_e1`.`run_e1` 재사용). ⚠미조사.
    'DROP:E2_EXTSYNTH_2026-09-03|E1_EXTBASE_2026-09-03|G-0',
    'DROP:E2_EXTSYNTH_2026-09-03|E1_EXTBASE_2026-09-03|G-ARM',
    'DROP:E2_EXTSYNTH_2026-09-03|E1_EXTBASE_2026-09-03|G-BUDGET',
    'DROP:E2_EXTSYNTH_2026-09-03|E1_EXTBASE_2026-09-03|G-DIR',
}


def check_gate_succession():
    """형제 승계 이름집합 래칫 — 새 소실이 들어오거나 기지 소실이 사라지면 실패."""
    r = subprocess.run([PY, 'tmp/_audit/check_gate_succession.py', '--keys'],
                       cwd=ROOT, capture_output=True, text=True)
    got = {l.split('\t', 1)[1] for l in r.stdout.split('\n') if l.startswith('KEY\t')}
    if r.returncode != 0 and not got:
        print('  [FAIL] 형제 승계 (래칫)')
        print('         ⛔검사기가 키를 내지 못했다 — 죽었거나 출력 형식이 바뀌었다.')
        for line in (r.stdout + r.stderr).strip().split('\n')[-6:]:
            print(f'         {line}')
        return False
    if got != GATE_SUCCESSION_KNOWN:
        print('  [FAIL] 형제 승계 (래칫)')
        for k in sorted(got - GATE_SUCCESSION_KNOWN):
            print(f'         ⛔새 소실: {k}  — 후속 배치의 사전등록에 **한 줄** 적어라'
                  ' ("`G-…` 는 …이므로 승계하지 않는다"). ⛔동결 배치는 고치지 말고'
                  ' 여기 등록 + 사유를 적어라.')
        for k in sorted(GATE_SUCCESSION_KNOWN - got):
            print(f'         ⚠기지 소실이 사라졌다: {k}  — 승계했는지, 배치가 개명/삭제됐는지 확인')
        return False
    print(f'  [OK  ] 형제 승계 (래칫: 기지 {len(got)}건 그대로)')
    return True


def check_frozen_sha256():
    import glob
    found = set()
    for f in sorted(glob.glob(os.path.join(ROOT, 'tmp', '*', 'FROZEN.sha256'))):
        d = os.path.dirname(f)
        r = subprocess.run(['sha256sum', '-c', 'FROZEN.sha256'], cwd=d,
                           capture_output=True)
        if r.returncode != 0:
            found.add(os.path.basename(d))
    new_bad, gone = found - SHA256_KNOWN_FAIL, SHA256_KNOWN_FAIL - found
    if new_bad or gone:
        print('  [FAIL] FROZEN.sha256 (래칫)')
        if new_bad:
            print(f'         ⛔새 실패 {len(new_bad)}: ' + ', '.join(sorted(new_bad)))
        if gone:
            print(f'         ⚠사라진 이름 {len(gone)}: ' + ', '.join(sorted(gone))
                  + '  (수리했다면 SHA256_KNOWN_FAIL 에서 **의도적으로** 지워라)')
        return False
    print(f'  [OK  ] FROZEN.sha256 (래칫: 기지 {len(found)}건 그대로)')
    return True


def check_batch_artifacts():
    """종료코드가 아니라 **문제 배치 이름 집합**을 본다."""
    p = subprocess.run([PY, 'tmp/_audit/check_batch_artifacts.py'],
                       cwd=ROOT, capture_output=True, text=True)
    out = (p.stdout + p.stderr).split('\n')
    try:
        i = next(k for k, l in enumerate(out) if '동결 증거 부재' in l)
    except StopIteration:
        print('  [FAIL] 배치 산출물 (래칫)')
        print('         ⛔검사기 출력에서 「동결 증거 부재」 절을 찾지 못했다 --- '
              '출력 형식이 바뀌었거나 검사기가 죽었다.')
        return False
    # ⚠접두어로 산문을 걸러내려 했더니 **`**판정 코드를…` 한 줄이 배치 이름으로 샜다**.
    #   ⟹ 구조로 판별한다: 배치 이름은 `tmp/` 아래 **실제로 존재하는 디렉터리**다
    #   (`VOID_unread` 만 예외 --- 검사기가 만드는 가상 항목).
    import re as _re
    found = set()
    for line in out[i + 1:]:
        t = line.strip()
        if not t:
            break
        if not _re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.\-]*', t):
            continue
        if t == 'VOID_unread' or os.path.isdir(os.path.join(ROOT, 'tmp', t)):
            found.add(t)
    new_bad = found - FROZEN_EVIDENCE_KNOWN
    gone = FROZEN_EVIDENCE_KNOWN - found
    if new_bad or gone:
        print('  [FAIL] 배치 산출물 (래칫)')
        if new_bad:
            print(f'         ⛔새 문제 배치 {len(new_bad)}: ' + ', '.join(sorted(new_bad)))
        if gone:
            print(f'         ⚠사라진 이름 {len(gone)}: ' + ', '.join(sorted(gone)))
            print('         (수리했다면 FROZEN_EVIDENCE_KNOWN 에서 **의도적으로** 지워라)')
        return False
    print(f'  [OK  ] 배치 산출물 (래칫: 기지 {len(found)}건 그대로)')
    return True


def run(name, cmd, verbose):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok = p.returncode == 0
    print(f'  [{"OK  " if ok else "FAIL"}] {name}')
    if not ok or verbose:
        tail = (p.stdout + p.stderr).strip().split('\n')
        for line in tail[-14:]:
            print(f'         {line}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='pytest 를 건너뛴다')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    print('── 방벽 ────────────────────────────────────────────────')
    checks = CHECKS + ([] if a.fast else SLOW)
    bad = [n for n, c in checks if not run(n, c, a.verbose)]
    if not check_batch_artifacts():
        bad.append('배치 산출물')
    if not check_frozen_sha256():
        bad.append('FROZEN.sha256')
    if not check_seed_bands():
        bad.append('시드 대역')
    if not check_mde_constants():
        bad.append('MDE 상수')
    if not check_gate_succession():
        bad.append('형제 승계')
    # ⚠[2026-09-02] **훅에서는 건너뛴다.** 실측 61 s 로 `--fast` 전체(93 s)를 지배했고,
    #   내가 이 파일에 스스로 적은 규칙(*"느린 훅은 사람이 `--no-verify` 로 우회한다"*)을
    #   어기고 있었다. ★경고 도구이므로 커밋마다 돌 필요가 없다 --- 전체 실행에는 남긴다.
    if not a.fast and not check_audit_suite():
        bad.append('감사 도구 묶음')
    print('────────────────────────────────────────────────────────')
    if bad:
        print(f'⛔실패 {len(bad)}: ' + ', '.join(bad))
        print('⚠래칫이 「개선」으로 실패했다면 기준선을 **의도적으로** 내려라 --- '
              '침묵시키지 마라.')
        return 1
    # ⚠래칫은 `CHECKS` 밖에서 부른다 — 배치산출물·FROZEN.sha256·시드대역·**MDE 상수**·
    #   **형제 승계**(+ `--fast` 가 아니면 감사 도구 묶음). 수를 손으로 세지 말고 여기서 더한다.
    print(f'✅{len(checks) + (5 if a.fast else 6)}종 전부 통과'
          + (' (pytest 건너뜀)' if a.fast else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
