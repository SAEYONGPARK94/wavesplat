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
# ⛔⛔[2026-09-03, 2차 감사가 초판 면제를 뚫었다 --- 세 방법 전부 실증됐다]
#   ① 표지가 **부분문자열**이라 평범한 낱말 `해당시점` 이 방벽을 껐다.
#   ② `continue` 가 줄 단위라 무관한 `당시` 가 앞 3줄에 있으면 그 줄 전체가 침묵했다.
#   ③ 전 줄에 표지를 달면 `[OK] 0곳 전부 일치` 로 **공허하게 초록**이었다.
#   ⟹ ①표지를 **명시 마커**로 바꾸고(산문에 우연히 나올 수 없다)
#     ②면제 건수를 **반드시 출력**하고 ③**커버리지 하한**을 건다.
HISTORICAL = ('<!--past-->',)
LOOKBACK = 0
MIN_COVERED = 2          # ⛔이보다 적게 검사되면 실패 --- 「전부 면제」를 막는다

# ★★[2026-09-03, 2차 감사의 가장 중요한 지적] 이 방벽은 **일치**만 강제하고
#   **진실**은 강제하지 않았다. 만든 계기인 `67a3eacd`(시험 22개 사고 삭제)가
#   **다시 일어나도 통과한다** --- 생성기가 줄어든 수를 쓰고, 방벽은 산문을 그 수로
#   **내리라고** 요구할 뿐이다. `CLAUDE.md` 자신이 *"시험 수가 줄면 그것은 신호다"* 라
#   적어 두었는데 대응하는 래칫이 없었다.
#   ⟹ **래칫**: 시험 수가 이 값보다 내려가면 실패한다. 의도적으로 줄였다면 **여기를
#     내리고 사유를 적어라**(저장소의 다른 래칫과 같은 규약 --- 침묵시키지 마라).
RATCHET = 268            # 2026-09-03 실측. ⛔내릴 때는 사유를 아래에 적는다.
# 이력: 233(09-03 오전, `67a3eacd` 사고 삭제 상태) → 264(복구) → 267(arm_provenance 3개)

# ★그리고 같은 값이 **두 곳에 독립 생성**된다(감사 지적):
#   `thesis/deploy_constants.tex \DeployNumTests` 와
#   `paper/paper_constants.tex \DeployNumTestsPaper` --- 아무도 둘을 대조하지 않았다.
#   ⟹ 아래에서 대조한다(저장소가 여섯 번 물린 「같은 값 두 곳」 부류).


def truth():
    m = re.search(r'\\newcommand\{\\DeployNumTests\}\{(\d+)\}',
                  open(CONST, encoding='utf-8').read())
    if not m:
        raise SystemExit('⛔`\\DeployNumTests` 를 찾지 못했다 --- 생성기를 먼저 돌려라')
    return int(m.group(1))


def paper_truth():
    """논문 쪽 독립 생성물. ⛔두 곳이 어긋나면 어느 쪽이 낡았는지 알 수 없다."""
    p = os.path.join(ROOT, 'paper', 'paper_constants.tex')
    if not os.path.exists(p):
        return None
    m = re.search(r'\\newcommand\{\\DeployNumTestsPaper\}\{(\d+)\}',
                  open(p, encoding='utf-8').read())
    return int(m.group(1)) if m else None


def main():
    n = truth()
    pn = paper_truth()
    if pn is not None and pn != n:
        print('── CLAUDE.md 시험 수 대조 ' + '─' * 32)
        print(f'  ⛔두 생성물이 어긋난다: thesis \\DeployNumTests={n} · '
              f'paper \\DeployNumTestsPaper={pn}')
        print('   ⚠생성기 둘을 다시 돌려라 --- 어느 쪽이 낡았는지는 이 검사기가 모른다.')
        return 1
    if n < RATCHET:
        print('── CLAUDE.md 시험 수 대조 ' + '─' * 32)
        print(f'  ⛔★래칫: 시험 수가 {RATCHET} → {n} 로 **줄었다**.')
        print('   ⚠사고 삭제인지 의도적 정리인지 이 검사기는 모른다 --- 확인하고,')
        print('     의도적이라면 `RATCHET` 을 내리고 **사유를 주석에 적어라**.')
        print('   ★선례: 2026-09-03 에 커밋 하나가 시험 22개를 조용히 지웠다.')
        return 1
    bad, seen, skipped = [], 0, []
    for p in CLAUDE:
        if not os.path.exists(p):
            continue
        lines = open(p, encoding='utf-8').read().split('\n')
        for i, line in enumerate(lines, 1):
            exempt = any(h in line for h in HISTORICAL)
            for m in PAT.finditer(line):
                if exempt:
                    skipped.append((os.path.relpath(p, ROOT), i, m.group(0)))
                    continue
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
    for f, i, txt in skipped:
        print(f'  [면제] {f}:{i}  "{txt.strip()}"  ← <!--past--> 마커')
    if seen < MIN_COVERED:
        print(f'\n⛔검사된 표기가 {seen}곳뿐이다(하한 {MIN_COVERED}). '
              '면제가 방벽을 비웠거나 정규식이 표기를 놓치고 있다.')
        return 1
    print(f'  [OK  ] 시험 수 표기 {seen}곳 전부 {n} 와 일치'
          + (f' · 면제 {len(skipped)}곳' if skipped else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
