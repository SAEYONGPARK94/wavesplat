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
  # ── 2026-08-28 추가 (M-4 · M-5b · M-8b) ────────────────────────────────
  "CAP-KR (용량 부호 반대)": "용량이모자라서지는것이아니다",
  "WIN-NOGAIN-KR (창 규약 무관 문장)": "창을줄여서제안표현이좋아진다는증거는어느읽기에서도없다",
  # ★2026-08-29 개명(M-8b AMENDMENT_2): "인수분해 불가" → "검출되지 않음".
  #   옛 앵커는 FORBIDDEN 으로 내려갔다(정정 문안이 금지 대상으로 *인용* 하므로 ⛔표지 앞뒤 확인).
  "LADDER-KR (원리적 검출 불가)": "가족mdeq=0.931",
  "LADDER-EN (family MDE)":      "family-wisemdeq=0.931",
  "LADDER-ADD (가법성 승수)":      "1.52074",
  # ── 2026-08-30 추가 (M-7): ★"체크포인트 = 평가 분포" 단서를 **약화** 시키는 절이
  #   저장소에 처음 생겼다. 그 단서는 논문 **열한 곳** 에 적혀 있으므로(초록 국/영 · §boundary ·
  #   기여 넷 · 제3장 넷 · 결론) **같은 문면** 이 그 곁에 있어야 한다 — 한 곳만 고치는 것이
  #   이 저장소가 반복해서 낸 "N곳 중 N-1곳" 결함이다.
  #   ⛔이 문면을 "닫혔다" 로 바꾸지 마라 — 약화이지 폐쇄가 아니다(FORBIDDEN 참조).
  "M7-WEAKEN-KR (약화 축자)": "단그단서는2026-08-30에약해졌다(닫히지않았다)---학습분포를실측앵커쪽으로의도적으로이동시킨두체크포인트가동결코퍼스에서확증문턱을유지한다(43/48및43/48;§).",
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
  # ★2026-08-30 (16인 심사 BLOCKER B-1): 제2장 "본 연구의 위치" 의 (ii) 가 척추와 정면
  #   충돌했다(*"그 경계를 만드는 원인을 목적함수 수준까지 분해한 것"*). 철회 문안은
  #   그 축자를 **재현하지 않으므로**(패러프레이즈로 적었다) 단순 부재 검사로 충분하다.
  "⛔철회 '경계를 만드는 원인을 … 분해' (B-1)":    "경계를만드는원인을목적함수수준까지분해",
}
# ★2026-08-29 신설(M-8b AMENDMENT_2): 개명으로 폐기된 문안. ⚠정정 문안이 이것을 **금지 대상으로
#   인용** 하므로 단순 부재 검사는 오탐이다 ⟹ `GUARDED` 는 앞뒤 60/80자에 ⛔·"철회" 표지가
#   없을 때만 FAIL 한다. (표지 없이 나오면 그것이 되감김이다.)
GUARDED = {
  "⛔개명 전 '단일 요인으로 인수분해되지 않' (M-8b)": "단일요인으로인수분해되지않",
  "⛔개명 전 '격차가 인수분해되지 않' (M-8b)":        "격차가인수분해되지않",
  "⛔개명 전 'does not factorise' (M-8b)":           "doesnotfactorise",
  # ★2026-08-30 (M-7): 되감김 감시 — E2 는 단서를 **약화** 시켰지 닫지 않았다.
  #   ⚠논문이 이 문면을 **금지 대상으로 인용** 하는 곳이 여럿이므로(⛔"…단서가 닫혔다") 단순
  #   부재 검사는 오탐이다 ⟹ GUARDED 에 둔다: 앞뒤에 ⛔·철회 표지가 없으면 그것이 되감김이다.
  "⛔되감김 '평가 분포 단서가 닫혔다' (M-7)":        "평가분포단서가닫혔다",
  "⛔되감김 '같은 분포라는 단서가 닫혔다' (M-7)":     "같은분포라는단서가닫혔다",
}
# ⚠`⟹ 상호작용`(M-8b 철회) 은 **여기서 검사하지 않는다** — `strip_tex` 가 `⟹` 를 지우므로
#   `B-18` 의 **사전 선언된 상호작용 검정**(thesis_main.tex 제3장, 정당한 용법)과 구별할 수 없다.
#   그 검사는 원문을 그대로 읽는 `tmp/M8b_LADDER_2026-08-28/factorization_check.py` 가 한다.

# ★2026-08-28 신설: **"같은 주장이 N곳이면 N곳 전부"** 를 기계로 강제한다.
#   사유 = 오늘 낸 판정 넷이 논문에 **0회** 였던 것이 위원장 사유 ④("문서의 지연")의
#   재발이었다. 한 곳만 고치는 결함은 canonical 근사일치로 잡히지만, **아예 빠진 곳**은
#   그 방식으로 안 잡힌다 ⟹ 여섯 지점(초록 국/영·기여·경계·제3장·결론)을 직접 센다.
#   ⛔여기 넣는 앵커는 **verdict 에서 나온 계수 원자**만 쓴다(문면은 바뀌어도 수는 남는다).
import bisect
def _line_span(startpat, endpat, after=0):
    s = e = None
    for i, l in enumerate(lines, 1):
        if i <= after: continue
        if s is None and re.search(startpat, l): s = i
        elif s is not None and re.search(endpat, l): e = i; break
    return s, (e or len(lines))

SPANS = {}
SPANS["초록KR"]   = _line_span(r'\\chapter\*\{초\\quad 록\}', r'\\chapter\*\{Abstract\}')
SPANS["초록EN"]   = _line_span(r'\\chapter\*\{Abstract\}', r'^\\newpage\s*$|tableofcontents')
SPANS["§boundary"] = _line_span(r'\\label\{sec:boundary\}', r'본 논문의 주요 기여는')
SPANS["기여"]     = _line_span(r'본 논문의 주요 기여는', r'\\section\{논문 구성\}')
SPANS["제3장"]    = _line_span(r'\\label\{chap:wavesplat\}', r'\\label\{chap:hydrofoil\}')
SPANS["결론"]     = _line_span(r'\\label\{sec:conclusion\}', r'\\section\{향후 연구\}')

COVERAGE = {
  "M-4 용량 (합성 0/48)":      "0/48",
  "M-5b 창 (제안 팔 2/12)":    "2/12",
  "M-8b 사다리 (끝점 11/12)":  "11/12",
  # ★2026-08-29(AMENDMENT_2): 가족 MDE 와 가법성 두 수도 **여섯 지점 전부** 에 있어야 한다.
  #   사유 = 이 두 가지가 빠지면 널이 다시 "인수분해 불가" 로 읽힌다(그 오독이 5곳에 복제됐다).
  "M-8b 가족 MDE (q=0.931)":   "0.931",
  "M-8b 가법성 격차 (0.217%)": "0.217",
  # ★2026-08-29 (M-1 · M-3): 오늘 낸 두 판정도 **여섯 지점 전부** 에 있어야 한다.
  #   사유 = 08-28 오전에 판정 넷을 논문에 넣지 않아 위원장 사유 ④("문서의 지연")가
  #   재발했다. 그 재발을 기계가 막는다.
  #   ⚠승수만으로는 부족하다 — "21/24"·"22/24" 는 B-26 표(제3장)에 **이미** 있어서
  #   제3장 칸이 M-1 없이도 통과한다 ⟹ **M-1 고유 크기(33.65)** 를 함께 요구한다.
  #   ⛔M-3 의 수심 셀 원자는 여기 넣지 않는다 — 처치 미집행(판정 무효)이라 인용 자체가 금지다.
  "M-1 비선형 (HOS 21/24)":    "21/24",
  "M-1 비선형 크기 (33.65%)":  "33.65",
  "M-1 선형 짝 (22/24)":       "22/24",
  "M-3 방향 셀 (13/16)":       "13/16",
  "M-3 방향폭 셀 (10/16)":     "10/16",
  # ★2026-08-30 (M-7): 저장소 최초의 **학습-측 개입**이다 ⟹ 여섯 지점 전부에 있어야 한다.
  #   ⚠"43/48" 은 단독으로 구별되지 않는다(B-12 표가 제3장에 이미 갖고 있다) ⟹ M-7 고유
  #     원자 둘을 **함께** 요구한다: 분포 주효과 p(0.0730)와 학습 시드 편차(9.14%).
  #   ⚠후자는 verdict 에 없다(레코드별 상대 크기는 판정 대상이 아니었다) ⟹ 단일출처는
  #     tmp/M7_SUPERVISION_2026-08-28/result_recheck.py 이고, 그 스크립트가 동결 jsonl 에서
  #     다시 만들어 **논문 본문과 직접 대조** 한다(= 손복사 방지의 같은 장치).
  #   ⛔E3(보고 전용)의 원자는 여기 넣지 않는다 — 확증 인용 자체가 금지다.
  "M-7 분포 이동 (DATA·BOTH 43/48)": "43/48",
  "M-7 실측 널 (분포 p=0.0730)":     "0.0730",
  "M-7 학습 시드 편차 (9.14%)":      "9.14",
  # ★2026-08-30 (M-10): 소거표의 "관측 기하" 행에서 **피복** 성분이 배제됐다.
  #   ⚠"45/48" 은 단독으로 구별되지 않는다(B-22 의 L2 가 같은 승수다) ⟹ 고유 크기를 쓴다.
  "M-10 피복 (obs:0_5 상대 중앙 -31.69%)": "31.69",
  # ★2026-08-30 (BLOCKER B-2): M-1 동결 드리프트 공시는 **인용 지점 전부**에 있어야 한다.
  #   문면은 preamble 매크로 하나가 정의하므로(= 단일출처) 여기서는 **매크로의 존재**를 센다.
  #   ⚠본문 정규화가 백슬래시와 중괄호만 지우므로 \MOneDrift{} 는 "MOneDrift" 로 남는다.
  #   ⟹ \MOneDriftFN / \MOneDrift / \MOneDriftEN 세 형태 전부가 이 접두어로 잡힌다.
  "M-1 동결 드리프트 공시 (매크로)": "MOneDrift",
}

def find_all(hay, needle):
    out, start = [], 0
    while True:
        k = hay.find(needle, start)
        if k < 0: break
        out.append(k); start = k + 1
    return out

_counts = {}
for name, needle in CANON.items():
    pos = find_all(flow, needle)
    _counts[name] = len(pos)
    print(f"\n== {name}\n   패턴({len(needle)}자) 완전일치 {len(pos)}회 → 줄 "
          f"{[idx[p] for p in pos]}")
    # 근사 일치: 같은 앵커 토큰을 가지나 문면이 다른 곳
    anchor = needle[:14]
    near = [p for p in find_all(flow, anchor) if p not in pos]
    for p in near:
        seg = flow[p:p+len(needle)+40]
        r = difflib.SequenceMatcher(None, needle, seg[:len(needle)]).ratio()
        print(f"   ⚠근사 :{idx[p]:<5} ratio={r:.2f}  …{seg[:110]}…")

# ═════════════════════════════════════════════════════════════════════════════
# ★2026-08-30 신설 — **출현 횟수 동결**. 돌연변이 시험이 구멍을 보였다:
#   척추 문장 `EXCL-SENTENCE` 를 **한 곳만** 개작하자 이 검사기는 `완전일치 3회 → 2회` 를
#   정확히 **찍었는데 exit 0** 이었다. ⟹ ★***"N곳 중 N-1곳" — 이 저장소가 스스로 누적 1위라
#   부르는 결함 — 을 검사기가 보고도 판정하지 않는다.*** run_all.sh:50 이 08-28 에 적어 둔
#   *"돌려도 전부 exit 0 이라 판정하지 않았다"* 가 같은 자리에서 재발한 것이다.
#   ★더 나쁜 것은 **기준선이 없다는 점**이다 — `2회` 라는 출력만 보고는 그것이 옳은지
#   알 수 없다(저장소 규율: *"동결한 적이 없으면 md5 열이 영원히 OK 를 찍는다"*).
# ⚠**판정으로 올리지 않고 자문으로 둔다(exit 0 유지).** 사유는 검출력이 아니라 **오탐**이다 —
#   2026-08-30 현재 척추 문안 자체가 개정 중이라(B-1·B-2 가 6곳·5곳을 의도적으로 고친다)
#   지금 실패로 올리면 **빨간불이 상시화돼 신호가 죽는다**. 개정이 끝난 뒤
#   `--refreeze` 하고, 그때 판정으로 승격할지 결정하라.
# ═════════════════════════════════════════════════════════════════════════════
import json
COUNTS_FILE = pathlib.Path("/home/vexit/wavesplat/wavesplat/tmp/STATUS_2026-08-28/thesis_canon_counts.json")
if "--refreeze" in sys.argv:
    COUNTS_FILE.write_text(json.dumps(_counts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n★출현 횟수 재동결: {len(_counts)}항목 → {COUNTS_FILE}")
    sys.exit(0)
print("\n--- 출현 횟수 대 동결 기준선 [자문] ---")
try:
    _frozen = json.loads(COUNTS_FILE.read_text(encoding="utf-8"))
except Exception:
    _frozen = None
    print("⚠기준선 파일이 없다 — `--refreeze` 로 만들어라. "
          "★기준선이 없으면 이 검사는 **영원히 통과**한다.")
if _frozen is not None:
    drift = [(n, _frozen.get(n), c) for n, c in _counts.items() if _frozen.get(n) != c]
    newly = [n for n in _counts if n not in _frozen]
    if not drift:
        print(f"  ok  {len(_counts)}항목 전부 동결값과 일치")
    for n, was, now in drift:
        arrow = "줄었다 ★N곳 중 N-1곳 결함 의심" if (was or 0) > now else "늘었다"
        print(f"  ⚠**{n}**: 동결 {was}회 → 현재 {now}회 ({arrow})")
    for n in newly:
        print(f"  ⚠미동결 항목(새로 추가됨): {n} = {_counts[n]}회")

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
# --- GUARDED (2026-08-29) — 표지 없는 되감김만 FAIL ---------------------------
print("\n--- GUARDED: 폐기 문안이 ⛔/철회 표지 없이 살아 있는가 ---")
_mark = re.compile(r"[⛔]|철회|retract", re.I)
for name, needle in GUARDED.items():
    pos = find_all(flow, needle)
    # ⚠flow 는 ⛔ 를 제거한 뒤의 문자열이다 ⟹ 원문 줄에서 표지를 본다.
    unguarded = []
    for p_ in pos:
        ln = idx[p_]
        win = "".join(lines[max(0, ln - 3):min(len(lines), ln + 3)])
        if not _mark.search(win):
            unguarded.append(ln)
    if unguarded:
        bad += len(unguarded)
        print(f"⛔FAIL {name}: 표지 없는 출현 {len(unguarded)}회 → 줄 {unguarded}")
    else:
        print(f"  ok  {name}: 출현 {len(pos)}회, 전부 표지 동반")

# --- COVERAGE (2026-08-28) --------------------------------------------------
print("\n--- COVERAGE: 같은 주장이 여섯 지점에 전부 있는가 ---")
def body_between(a, b):
    out = []
    for i in range(a, b + 1):
        l = lines[i-1]
        if l.lstrip().startswith('%'): continue
        out.append(re.sub(r'(?<!\\)%.*$', '', l))
    return re.sub(r'[{}$\\]|\s+', '', "".join(out))

for name, atom in COVERAGE.items():
    missing = [k for k, (a, b) in SPANS.items() if a is None or atom not in body_between(a, b)]
    if missing:
        bad += len(missing)
        print(f"⛔FAIL {name}: 빠진 지점 {missing}")
    else:
        print(f"  ok  {name}: 여섯 지점 전부")

n_spine = len(find_all(flow, CANON["SPINE-KR (단일 진술 축자)"]))
if n_spine < 5:
    print(f"⚠경고 SPINE-KR 이 {n_spine}회다(동결 시점 5회) --- "
          "★'N곳 중 N-1곳' 결함일 수 있으니 위 근사 목록을 읽어라. (경고일 뿐 실패는 아니다)")
print(f"\n금지 문면 {bad}건")
sys.exit(1 if bad else 0)
