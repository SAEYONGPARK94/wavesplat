#!/usr/bin/env python3
"""손복사 기준 상수 사냥꾼 — **두 번째 기계적 방벽** (2026-08-23).

오늘 이 부류가 **세 번** 물었다:
  · `gates_b2.ZERO_REF = 1.0`   — 정의상 참이 아니었다(실측 0.3375–1.1559) ⟹ 실측 아크 전 계수 철회
  · `gates_w13/w14.ZERO_REF = 0.25` — 원리적으로 틀렸다(실측 0.180–0.292; 다행히 이동 ≤1/192)
  · `W10` 의 `R_BASE` 손복사 반올림(1.5163 vs 1.5162676851150798) ⟹ 게이트 FAIL

**정의**: *"기준선·기저·문턱을 모듈 수준 스칼라 리터럴로 적어 두고, 그 값을 자료에서
재도출하지 않는 것."* 자료가 바뀌면 조용히 틀리고, **틀린 채로 게이트를 통과한다.**

이 스크립트는 `tmp/**/gates*.py` 를 훑어 **의심 상수**를 뽑는다. 판정하지 않는다 —
**사람이 볼 목록을 만든다.** 옆에 `def <같은이름>()` 이 있거나 주석에 `재계산`/`recompute`/
`원본` 이 있으면 '유래 있음' 으로 낮춘다.

사용:  python3 tmp/_audit/find_handcopied_refs.py
"""
import glob, os, re, sys

# ★두 부류를 가른다 (이 구분이 없으면 도구가 잡음이 된다):
#   DATA  = **자료에서 나와야 하는 양**(기준선·기저·제로필드). 손복사면 자료가 바뀔 때 조용히 틀린다.
#   RULE  = **사전등록한 결정 규칙**(문턱 카운트). 리터럴이 **옳다** — 결과 전에 고정하는 것이 목적이다.
DATA = re.compile(
    r'^\s*(?P<name>[A-Z][A-Z0-9_]*(?:REF|BASE|ZERO|BASELINE|FLOOR)[A-Z0-9_]*)'
    r'\s*=\s*(?P<val>-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(?P<rest>#.*)?$')
RULE = re.compile(
    r'^\s*(?P<name>[A-Z][A-Z0-9_]*(?:THRESH|THRESHOLD|NOMINAL|CONFIRM)[A-Z0-9_]*)'
    r'\s*=\s*(?P<val>-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(?P<rest>#.*)?$')
SUSPECT = DATA
DERIVED = re.compile(r'재계산|recompute|원본에서|from source|자료에서|derive')


def main():
    hits = []
    for f in sorted(glob.glob('tmp/**/gates*.py', recursive=True)):
        src = open(f).read()
        lines = src.splitlines()
        for i, ln in enumerate(lines, 1):
            m = SUSPECT.match(ln)
            if not m:
                continue
            name, val, rest = m.group('name'), m.group('val'), m.group('rest') or ''
            ctx = '\n'.join(lines[max(0, i - 4):i + 1])
            derived = bool(DERIVED.search(ctx)) or f'def {name.lower()}(' in src
            hits.append((f, i, name, val, derived, rest.strip()[:60]))
    bad = [h for h in hits if not h[4]]
    rules = []
    for f in sorted(glob.glob('tmp/**/gates*.py', recursive=True)):
        for i, ln in enumerate(open(f).read().splitlines(), 1):
            m = RULE.match(ln)
            if m: rules.append((os.path.relpath(f), i, m.group('name'), m.group('val')))
    print(f"{'파일':46s} {'줄':>5s} {'이름':22s} {'값':>10s}  유래")
    for f, i, n, v, d, c in hits:
        print(f"{os.path.relpath(f)[-46:]:46s} {i:5d} {n[:22]:22s} {v:>10s}  "
              f"{'유래있음' if d else '⚠손복사 의심'}  {c}")
    print(f"\n손복사 의심(DATA 부류): {len(bad)}/{len(hits)}")
    print(f"참고 — 사전등록 결정 규칙(RULE 부류, 리터럴이 옳다): {len(rules)}건")
    print("⚠교훈: `gates_b2.py:21` 은 주석에 **'손복사 아님'** 이라 적혀 있었고 **그게 틀렸다.**"
          " 주석은 증거가 아니다 — 자료를 통과시켜라.")
    print("★처방: 기준선을 상수로 선언하지 말고 **채점 함수에 영(zero) 예측을 통과시켜** "
          "자료에서 산출하라. 못 하면 최소한 원본에서 재계산해 `assert` 로 대조하라.")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
