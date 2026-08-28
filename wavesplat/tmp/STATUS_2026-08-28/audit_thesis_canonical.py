#!/usr/bin/env python3
"""논문 정합 감사 #2 — **동일 주장 다중 지점**의 축자 일치 검사.

런 0회. 각 CANON 항목에 대해 (a) 정규화 후 완전일치 지점 (b) 정규화 후 근사일치
(핵심 토큰은 있으나 문면이 다른) 지점을 나눠 보고한다. 후자가 "N곳 중 N-1곳" 결함이다.
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_canonical.py
"""
import re, sys, hashlib, pathlib, difflib

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
flow = "".join(flow).lower()   # ★영문 패턴은 소문자 고정 — 대소문자로 침묵하지 않도록

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
  "NAMED-AXIS-NEW (신 문안)": "배제된축하나를이름붙였을뿐",
  "EXCL-SENTENCE (배제 축자)": "경계는파랑조건의축을따라놓여있지않다",
  "M2-DISCLOSE (공시 축자)": "그승수에는공시둘이동반된다",
}

# ★2026-08-28 신설: **되감김 형태**를 보고가 아니라 실패로 승격한다.
#   사유 = 이 검사기는 08-28 오전에 만들어졌는데, 같은 날 오후 척추가 두 곳에서
#   되감겼을 때 `NAMED-AXIS-OLD 1회` 를 **찍고도 exit 0** 이라 아무도 읽지 않았다.
#   ⟹ 기계가 있는데도 재발한 이유는 "검사하지 않아서" 가 아니라 **"판정하지 않아서"** 다.
#   ⛔여기에는 **오탐이 없는 것만** 넣는다(전부 정정으로 폐기된 축자 문면이다).
FORBIDDEN = {
  "⛔구 문안 '축을 이름 붙였을 뿐' (지목)":        "축을이름붙였을뿐",
  "⛔되감김 '한 겹 = 경계가 놓인 축이다' (지목)":  "경계가놓인축이다",
  "⛔되감김 '경계가 놓인 축을 보고한다' (지목)":   "경계가놓인축을보고한다",
  "⛔지목형 '경계는 …축을 따라 놓여 있다'":        "축을따라놓여있다",
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

# --- 판정 -------------------------------------------------------------------
bad = 0
print("\n" + "=" * 60)
for name, needle in FORBIDDEN.items():
    pos = find_all(flow, needle)
    if pos:
        bad += len(pos)
        print(f"⛔FAIL {name}: {len(pos)}회 → 줄 {[idx[p] for p in pos]}")
    else:
        print(f"  ok  {name}")
n_spine = len(find_all(flow, CANON["SPINE-KR (단일 진술 축자)"]))
if n_spine < 5:
    print(f"⚠경고 SPINE-KR 이 {n_spine}회다(동결 시점 5회) --- "
          "★'N곳 중 N-1곳' 결함일 수 있으니 위 근사 목록을 읽어라. (경고일 뿐 실패는 아니다)")
print(f"\n금지 문면 {bad}건")
sys.exit(1 if bad else 0)
