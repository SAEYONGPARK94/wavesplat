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
 ("소거된 후보 수(개입)", r'각각 개입으로 시험|tested by intervention',     r'(네|셋|넷|세|\d)\s*(?:개|후보|가지)?'),
 ("B-26 에피소드 수",     r'B-26|SS-REAL|해상상태',                      r'(\d{2,3})\s*에피소드'),
 ("제4장(제어) 지위",     r'제어 응용|control application|하이드로포일 제어', r'(기여로 주장하지 않|does not claim|기여)'),
 ("실측 자기시험 최소",   r'자기 ?시험|self-test|RealSelfTestMin',        r'(\d+)'),
 ("지연 배율",            r'1\.5|2\.6|배율',                            r'(1\.5--2\.6|1\.5-2\.6)'),
 ("렌더 예산",            r'137|720',                                  r'(137|720)'),
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
