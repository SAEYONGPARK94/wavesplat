#!/usr/bin/env python3
"""★`CLAUDE.md` 의 **시험 수**를 생성 상수와 대조한다 (2026-09-03, 방벽 28번).

**왜 만들었나.** 오늘 재검이 `CLAUDE.md` 안에 시험 수가 **세 값**으로 공존하는 것을 찾았다
--- `250 tests`(11행) · `250 passed`(47행) · `264 tests`(64행), 그리고 **실측은 267**.
더 나쁜 것은 그날의 「250 → 264 정정」 커밋이 **세 곳 중 하나만** 고쳤다는 사실이다.
이 저장소가 스스로 목록으로 적어 둔 **N−1 결함 부류**가 *정정 커밋 안에서* 재발했다.

**왜 이 형태인가.** 논문·학위논문의 같은 수는 이미 `gen_deploy_constants.py` 가 만든다.
산문 파일은 생성 대상이 아니므로 **생성 상수와 대조하는 검사기**가 그 자리를 대신한다.
⟹ ***손으로 적을 수는 있지만, 틀린 채로는 커밋되지 않는다.***

⛔이 검사기는 시험을 **돌리지 않는다**(느리다). 생성 상수를 정본으로 삼고,
그 상수 자신은 `gen_deploy_constants.py` 가 실측으로 채운다.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLAUDE = [os.path.join(ROOT, '..', 'CLAUDE.md'), os.path.join(ROOT, 'CLAUDE.md')]
CONST = os.path.join(ROOT, 'thesis', 'deploy_constants.tex')

PAT = re.compile(r'\*?\*?(\d{2,4})\*?\*?\s*(tests|passed)\b')

# ⚠**과거 기록 면제** --- `CLAUDE.md` 는 옛 측정을 「그 당시의 기록」이라 명시하고 보존한다
#   (예: 2026-08-13 의 `111 passed`). 그것은 **현재 주장이 아니라 사료**이므로 대조하지 않는다.
#   ⛔면제는 **좁게** 잡는다: 같은 줄 또는 **바로 위 3줄**에 아래 표지가 있어야 한다.
#   ★면제 자체가 구멍이 될 수 있어 표지를 「당시」로 한정했다 --- 현재 주장에는 쓰지 않는 말이다.
HISTORICAL = ('당시', '그때의 기록')
LOOKBACK = 3


def truth():
    m = re.search(r'\\newcommand\{\\DeployNumTests\}\{(\d+)\}',
                  open(CONST, encoding='utf-8').read())
    if not m:
        raise SystemExit('⛔`\\DeployNumTests` 를 찾지 못했다 --- 생성기를 먼저 돌려라')
    return int(m.group(1))


def main():
    n = truth()
    bad, seen = [], 0
    for p in CLAUDE:
        if not os.path.exists(p):
            continue
        lines = open(p, encoding='utf-8').read().split('\n')
        for i, line in enumerate(lines, 1):
            ctx = '\n'.join(lines[max(0, i - 1 - LOOKBACK):i])
            if any(h in ctx for h in HISTORICAL):
                continue
            for m in PAT.finditer(line):
                seen += 1
                if int(m.group(1)) != n:
                    bad.append((os.path.relpath(p, ROOT), i, m.group(0)))
    print('── CLAUDE.md 시험 수 대조 ' + '─' * 32)
    if bad:
        for f, i, txt in bad:
            print(f'  ⛔{f}:{i}  "{txt.strip()}"  ← 생성 상수는 {n}')
        print(f'\n⛔{len(bad)}건 --- `\\DeployNumTests`={n} 와 어긋난다.')
        print('   ⚠수를 바꿨다면 `scripts/gen_deploy_constants.py` 를 먼저 돌려라')
        print('     (그것이 실측한다). ⛔산문만 고치면 이 검사기가 다시 막는다.')
        return 1
    print(f'  [OK  ] 시험 수 표기 {seen}곳 전부 {n} 와 일치')
    return 0


if __name__ == '__main__':
    sys.exit(main())
