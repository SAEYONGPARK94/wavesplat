#!/usr/bin/env python3
"""논문 정합 감사 #1 — 주장 원자(계수·크기·p) 추출과 다중 출현 대조.

런 0회. thesis_main.tex 만 읽는다. 아무것도 고치지 않는다.
출력: 같은 값이 여러 곳에 나오는 지점 전부 + 그 문맥(주석/본문 구분).
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_claims.py [--only PATTERN]
"""
import re, sys, collections, hashlib, pathlib

TEX = pathlib.Path("/home/vexit/wavesplat/wavesplat/thesis/thesis_main.tex")
raw = TEX.read_text(encoding="utf-8").splitlines()
print(f"# file md5 = {hashlib.md5(TEX.read_bytes()).hexdigest()}  lines = {len(raw)}\n")

def is_comment(l):           # 줄 전체 주석
    return l.lstrip().startswith('%')

# --- 원자 패턴 -------------------------------------------------------------
PATS = {
    "count":  re.compile(r'\$?\\?mathbf\{?\$?(\d{1,3})\}?\s*/\s*\{?(\d{1,3})\}?\$?'),
    "frac":   re.compile(r'(\d{1,3})\s*/\s*(\d{1,3})'),
    "pct":    re.compile(r'([+-]?\d{1,3}(?:\.\d{1,3})?)\s*\\%'),
    "pval":   re.compile(r'p\s*(?:=|＝)\s*\$?([\d.]+)\s*(?:\\times\s*10\^\{(-?\d+)\})?'),
    "sci":    re.compile(r'([\d.]+)\s*\\times\s*10\^\{(-?\d+)\}'),
}

hits = collections.defaultdict(list)   # value -> [(lineno, kind, context)]
for i, line in enumerate(raw, 1):
    body = line
    kind_prefix = "CMT " if is_comment(line) else "BODY"
    # 줄 안 %-주석 꼬리는 본문에서 제외
    if not is_comment(line):
        body = re.sub(r'(?<!\\)%.*$', '', line)
    for name, pat in PATS.items():
        for m in pat.finditer(body):
            if name == "frac":
                a, b = int(m.group(1)), int(m.group(2))
                if b == 0 or a > b or b > 600:   # 계수처럼 보이는 것만
                    continue
                key = f"{a}/{b}"
            elif name == "count":
                continue
            elif name == "pct":
                key = f"{float(m.group(1)):+.2f}%"
            elif name == "pval":
                mant = m.group(1); ex = m.group(2)
                key = f"p={mant}e{ex}" if ex else f"p={mant}"
            else:
                key = f"{m.group(1)}e{m.group(2)}"
            hits[(name, key)].append((i, kind_prefix, line.strip()[:150]))

only = None
if "--only" in sys.argv:
    only = sys.argv[sys.argv.index("--only") + 1]

for (name, key), locs in sorted(hits.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    if only and only not in key:
        continue
    body_locs = [l for l in locs if l[1] == "BODY"]
    if only is None and len(locs) < 2:
        continue          # 단일 출현은 대조 대상이 아니다
    print(f"== [{name}] {key}   (총 {len(locs)}회, 본문 {len(body_locs)}회)")
    for ln, kp, txt in locs:
        print(f"   {kp} :{ln:<5} {txt}")
    print()
