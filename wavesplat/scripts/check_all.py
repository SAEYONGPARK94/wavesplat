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
    # ⚠배치 검사기는 **상시 rc=1** 이다 --- 역사적 배치 넷이 동결 증거를 못 갖췄고
    #   그건 지금 고칠 수 있는 것이 아니다(런이 이미 끝났다).
    #   ⛔상시 빨간 검사는 사람이 **무시하기 시작한다** ⟹ 종료코드가 아니라
    #   **문제 배치의 이름 집합**을 래칫한다(수 아니라 이름 = 교체를 못 숨긴다).
]
SLOW = [('pytest', [PY, '-m', 'pytest', 'wavesim/tests/', 'wavesplat/tests/', '-q'])]

# ★2026-09-02 실측. 이 넷은 **런이 이미 끝난** 배치라 소급 수리가 불가능하다.
#   ⛔새 이름이 들어오면 실패한다 --- 그게 이 래칫의 전부다.
#   ⛔이름이 **빠져도** 실패한다: 배치가 조용히 사라지거나 개명된 것이므로 확인해야 한다.
FROZEN_EVIDENCE_KNOWN = {
    'B2_REAL_2026-08-23',
    'M1_NONLIN_REAL_2026-08-28',
    'M4_CAPACITY_2026-08-28',
    'VOID_unread',
}


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
    print('────────────────────────────────────────────────────────')
    if bad:
        print(f'⛔실패 {len(bad)}: ' + ', '.join(bad))
        print('⚠래칫이 「개선」으로 실패했다면 기준선을 **의도적으로** 내려라 --- '
              '침묵시키지 마라.')
        return 1
    print(f'✅{len(checks) + 1}종 전부 통과'
          + (' (pytest 건너뜀)' if a.fast else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
