#!/usr/bin/env python3
"""논문 정합 감사 #5 — **자기모순 사냥**: 같은 사실을 두 곳이 다르게 말하는가.

런 0회. 각 항목은 "이 사실을 말하는 모든 줄" 을 모아 값이 갈리는지 본다.
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_contradictions.py
"""
import re, hashlib, pathlib, collections

TEX = pathlib.Path("/home/vexit/wavesplat/wavesplat/thesis/thesis_main.tex")
lines = TEX.read_text(encoding="utf-8").splitlines()
print(f"# tex md5 = {hashlib.md5(TEX.read_bytes()).hexdigest()}\n")
BODY = [(i, re.sub(r'(?<!\\)%.*$', '', l)) for i, l in enumerate(lines, 1)
        if not l.lstrip().startswith('%') and l.strip()]

# (사실 이름, 그 사실을 말하는 줄을 찾는 정규식, 그 줄에서 값을 뽑는 정규식)
FACTS = [
 ("실측 레코드 수",       r'실측.{0,12}레코드|real records|고정 관측소',    r'(\d{1,3})\s*(?:개)?\s*레코드|(\w+)\s+records'),
 ("실측 사이트 수",       r'사이트',                                    r'사이트[^\d]{0,6}(\d+)|(\d+)\s*(?:곳|사이트)'),
 ("배포 테스트 수",       r'테스트|test suite|\\DeployNumTests',          r'(\d{3})\s*(?:개)?\s*(?:테스트|tests)'),
 ("소거 후보 남은 수",    r'남은 후보|아직 측정되지 않은 후보|remaining candidate', r'(다섯|넷|셋|\d)'),
 ("소거된 후보 수(개입)", r'각각 개입으로 시험|tested by intervention',     r'(다섯|네|셋|넷|세|\d)\s*(?:개|후보|가지)?'),
 # ── 2026-08-28 추가 (M-4·M-5b·M-8b) ─────────────────────────────────────────
 # ①소거표의 "부호가 반대" 행이 몇 개라 말하는가 ②사다리 고리 수 ③남은 축 서술의 개수
 ("부호 반대 행 수",      r'부호가 반대|opposite of the one predicted',   r'(둘|두|하나|한)\s*(?:다|행|후보)?'),
 ("사다리 고리 수",       r'고리|rung',                                 r'(여섯|six|\d)\s*(?:고리|rungs?)'),
 ("남은 축 개수",         r'남은 후보는|아직 측정되지 않은 후보|remaining candidate', r'(다섯|넷|셋|둘|\d)\s*(?:인데|이다|개)?'),
 # ★2026-08-29(AMENDMENT_2): MDE 는 **두 값** 이다 — 단일검정 0.869 · 가족 Holm 0.931.
 #   ⟹ 한 항목으로 묶으면 정상 상태가 '⚠갈림' 으로 찍힌다. **문맥별로 갈라** 놓아야
 #   *바꿔 붙인 경우* 가 갈림으로 드러난다(그것이 이 개정이 고친 결함이다).
 #   ⚠값 추출은 **문맥에 붙여서** 한다 — 한 줄이 두 값을 대비시키는 문장이 여럿이라
 #   ("단일 … 0.869 --- ⛔가족 값 0.931 과 섞지 않는다") 줄 단위로는 못 가른다.
 ("M-8b MDE(단일검정=PRIMARY)", r'0\.869|0\.931',
                               r'(?:단일 ?검정|single-test)[^0-9]{0,80}?(0\.\d{3})'),
 ("M-8b MDE(가족 Holm)",        r'0\.869|0\.931',
                               r'(?:가족(?:단위)?|family-wise)[^0-9]{0,80}?(0\.\d{3})'),
 ("B-26 에피소드 수",     r'B-26|SS-REAL|해상상태',                      r'(\d{2,3})\s*에피소드'),
 ("제4장(제어) 지위",     r'제어 응용|control application|하이드로포일 제어', r'(기여로 주장하지 않|does not claim|기여)'),
 ("실측 자기시험 최소",   r'자기 ?시험|self-test|RealSelfTestMin',        r'(\d+)'),
 ("지연 배율",            r'1\.5|2\.6|배율',                            r'(1\.5--2\.6|1\.5-2\.6)'),
 # ⚠2026-08-30 정정(오탐 제거, ⛔검사 범위는 좁히지 않는다): 종전 패턴은 **자릿수 경계를
 #   보지 않아** `0.1373`(M-7 의 배포 체크포인트 평균)을 렌더 예산으로 셌다. 줄을 고르는
 #   패턴은 **그대로 두고** 값 추출에만 경계를 붙인다 ⟹ 세는 대상은 줄지 않고 오탐만 준다.
 ("렌더 예산",            r'(?<![\d.])(?:137|720)(?![\d.])',           r'(?<![\d.])(137|720)(?![\d.])'),
]
for name, findpat, valpat in FACTS:
    fx, vx = re.compile(findpat), re.compile(valpat)
    vals = collections.defaultdict(list)
    for i, b in BODY:
        if not fx.search(b):
            continue
        for m in vx.finditer(b):
            v = next((g for g in m.groups() if g), None)
            if v:
                vals[v].append(i)
    if not vals:
        continue
    multi = len(vals) > 1
    print(f"{'⚠갈림' if multi else '  단일'} {name}: " +
          " · ".join(f"'{k}'×{len(v)}{v[:4]}" for k, v in sorted(vals.items(), key=lambda kv: -len(kv[1]))[:6]))
