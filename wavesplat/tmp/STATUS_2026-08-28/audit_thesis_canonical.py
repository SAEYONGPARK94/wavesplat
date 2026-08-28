#!/usr/bin/env python3
"""논문 정합 감사 #2 — **동일 주장 다중 지점**의 축자 일치 검사.

런 0회. 각 CANON 항목에 대해 (a) 정규화 후 완전일치 지점 (b) 정규화 후 근사일치
(핵심 토큰은 있으나 문면이 다른) 지점을 나눠 보고한다. 후자가 "N곳 중 N-1곳" 결함이다.
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_canonical.py
"""
import re, hashlib, pathlib, difflib

TEX = pathlib.Path("/home/vexit/wavesplat/wavesplat/thesis/thesis_main.tex")
raw = TEX.read_text(encoding="utf-8")
print(f"# md5 = {hashlib.md5(TEX.read_bytes()).hexdigest()}")
lines = raw.splitlines()

def strip_tex(s):
    s = re.sub(r'\\(?:textbf|emph|texttt|mathbf|textit)\s*\{', '{', s)
    s = re.sub(r'\\S\\?ref\{[^}]*\}', '§', s)
    s = re.sub(r'\\(?:ref|label|S)\s*\{?[^}\s]*\}?', '', s)
    s = re.sub(r'[{}$\\~]', '', s)
    s = re.sub(r'[★⛔⚠✅⟹%]', '', s)
    s = re.sub(r'\s+', '', s)
    return s

# 줄 전체 주석 제외한 본문을 하나의 흐름으로 (줄번호 매핑 유지)
flow, idx = [], []
for i, l in enumerate(lines, 1):
    if l.lstrip().startswith('%'):
        continue
    body = re.sub(r'(?<!\\)%.*$', '', l)
    for ch in strip_tex(body):
        flow.append(ch); idx.append(i)
flow = "".join(flow)

CANON = {
  "SPINE-KR (단일 진술 축자)":
     "본논문은작동경계의위치를보고하고,파랑조건축을배제하되,그경계가놓인축과그축위의원인은제시하지않는다",
  "SPINE-EN (영문 미러)":
     "reportswheretheoperatingboundaryliesandexcludesthewave-conditionaxis,whileidentifyingneithertheaxisitliesalongnoranycausealongit",
  "EXCL-KR (배제는 지목이 아니다)": "배제는지목이아니다",
  "NORANK-KR (코너 순위 불주장)": "코너사이의순위는주장하지않는다",
  "NORANK-KR2 (짧은 형태)": "코너순위는주장하지않는다",
  "NORANK-EN": "norankingamongthecellsisclaimed",
  "WIN-KR (승수가 가장 높다)": "승수가가장높다",
  "NAMED-AXIS-OLD (⛔구 문안)": "축을이름붙였을뿐",
  "NAMED-AXIS-NEW (신 문안)": "배제된축하나를이름붙였을뿐",
}

def find_all(hay, needle):
    out, start = [], 0
    while True:
        k = hay.find(needle, start)
        if k < 0: break
        out.append(k); start = k + 1
    return out

for name, needle in CANON.items():
    pos = find_all(flow, needle)
    print(f"\n== {name}\n   패턴({len(needle)}자) 완전일치 {len(pos)}회 → 줄 "
          f"{[idx[p] for p in pos]}")
    # 근사 일치: 같은 앵커 토큰을 가지나 문면이 다른 곳
    anchor = needle[:14]
    near = [p for p in find_all(flow, anchor) if p not in pos]
    for p in near:
        seg = flow[p:p+len(needle)+40]
        r = difflib.SequenceMatcher(None, needle, seg[:len(needle)]).ratio()
        print(f"   ⚠근사 :{idx[p]:<5} ratio={r:.2f}  …{seg[:110]}…")
