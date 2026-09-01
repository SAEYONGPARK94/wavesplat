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
    ('상수 생성기 --check', [PY, 'scripts/gen_paper_constants.py', '--check']),
    ('동반 공시', [PY, 'scripts/check_paper_disclosures.py']),
    ('★항진 검사', [PY, 'scripts/check_paper_disclosures.py', '--selftest']),
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
    # ⛔[2026-09-02] **학위논문 상수 생성기가 방벽에 없었다** ⟹ 조용히 표류했다
    #   (오늘 테스트를 둘 추가하자 `\DeployNumTests` 가 261 로 낡았다).
    #   ★재생성이 곧 검사다 --- git 이 차이를 보여준다.
    ('학위 상수 재생성', [PY, 'scripts/gen_deploy_constants.py']),
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


AUDIT_WARN_BASELINE = 35   # ★2026-09-02 실측. 늘어나면 실패, 줄어도 실패(기록 강제).
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
    print(f'✅{len(checks) + (2 if a.fast else 3)}종 전부 통과'
          + (' (pytest 건너뜀)' if a.fast else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
