#!/usr/bin/env python3
"""논문 정합 감사 #3 — 인용금지 문구/수치의 기계 대조.

런 0회. reference_citation_bans.md 의 08-22~08-28 묶음에서 **손으로 옮기지 않고**
패턴을 정의하되, 각 패턴은 금지목록의 해당 줄을 근거로 함께 싣는다.
BAN_NUM 은 "본문에 이 수가 있으면 반드시 동반돼야 할 단서" 를 함께 검사한다.
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_bans.py
"""
import re, hashlib, pathlib

TEX  = pathlib.Path("/home/vexit/wavesplat/wavesplat/thesis/thesis_main.tex")
BANS = pathlib.Path("/home/vexit/.claude/projects/-home-vexit-wavesplat/memory/reference_citation_bans.md")
lines = TEX.read_text(encoding="utf-8").splitlines()
bantxt = BANS.read_text(encoding="utf-8")
print(f"# tex md5 = {hashlib.md5(TEX.read_bytes()).hexdigest()}")
print(f"# bans md5 = {hashlib.md5(BANS.read_bytes()).hexdigest()}\n")

def body_lines():
    for i, l in enumerate(lines, 1):
        if l.lstrip().startswith('%'):
            continue
        b = re.sub(r'(?<!\\)%.*$', '', l)
        if b.strip():
            yield i, b

BODY = list(body_lines())
def norm(s):
    s = re.sub(r'\\(?:textbf|emph|texttt|mathbf|textit)\s*\{', '{', s)
    s = re.sub(r'[{}$\\]', '', s)
    return re.sub(r'\s+', '', s)
FLAT = [(i, norm(b), b) for i, b in BODY]

# --- (A) 금지 문구: 정규식 + 금지목록 근거 줄 ---------------------------------
PHRASE = [
 ("⛔코너 강도/순위 (08-28 F1)", r'가장(강|셈|세)|칸이가장|strongestcell|thestrongest'),
 ("⛔'해상상태가 무관'", r'해상상태가무관|해상상태는무관|seastateisirrelevant'),
 ("⛔원인 지목 '관측 모형이 원인'", r'원인은?관측모형(이다|입니다)|원인이관측모형|causeistheobservationmodel'),
 ("⛔'희소 −93%'", r'-?93\\?%|93\\?%감소'),
 ("⛔'85/100'·'84/100'", r'8[45]/100'),
 ("⛔'0.223'/'0.319'/'13.4'ms/'132.6'ms 헤드라인", r'0\.223|0\.319|13\.4\\?,?ms|132\.6'),
 ("⛔'약 10배' 지연", r'약10배|10배빠'),
 ("⛔'All phases complete'", r'Allphasescomplete|모든단계완료'),
 ("⛔'재학습이 다음 축' (08-28 B-23)", r'재학습.{0,12}(다음|우선|1순위|처방)'),
 ("⛔'B-20b 독립 확증/완전 blind/복제'", r'독립확증|완전blind|완전블라인드|B-20b.{0,20}복제'),
 ("⛔S13 헤드라인 '56/64'·'61/64'·'−28.2%'", r'56/64|61/64|-28\.2\\?%'),
 ("⛔'−23.96%' 정확도 크기", r'-?23\.96'),
 ("⛔'PROPAGATION-EARNS'/'DISPERSION-EARNS'", r'PROPAGATION-EARNS|DISPERSION-EARNS'),
 ("⛔'US 12,054,225' 를 파도장 특허로", r'12,?054,?225'),
 ("⛔'ZERO_REF=1.0' 기준 실측 계수", r'ZERO_?REF.{0,4}1\.0'),
]
for name, pat in PHRASE:
    rx = re.compile(pat, re.I)
    hit = [(i, b) for i, f, b in FLAT if rx.search(f)]
    tag = "⛔HIT" if hit else "  ok "
    print(f"{tag} {name}: {len(hit)}건")
    for i, b in hit[:12]:
        print(f"        :{i:<5} {b.strip()[:130]}")

# --- (B) 단서 동반 의무: 수치 → 같은 절 안에 반드시 있어야 할 앵커 ------------
print("\n--- (B) 단서 동반 의무 (수치가 나오는 곳의 ±60줄 창) ---")
DUTY = [
 ("SS-REAL 승수 24/24 ⟹ F2(국소화 반경 고정) 공시 동반 의무",
  r'24/24', r'국소화반경|loc_radius|loc/|Gaspari'),
 ("−42.42% ⟹ F2 공시 동반 의무", r'-42\.42', r'국소화반경|loc_radius|Gaspari'),
 ("'네 칸 유지/헤드라인이 산다' ⟹ 선형 참값 단서 동반 의무",
  r'네칸이모두유지|네칸이유지|allfourcellsholding', r'선형|linear'),
 ("−38.840%/48/48(B-20b) ⟹ blind 범위·계수기 공백 단서",
  r'-38\.84', r'blind|독립|사고|공백|계수기|replaces'),
 ("SS-LOWH 17/24 ⟹ EXPLORATORY/명목 병기", r'17/24', r'명목|탐색적|EXPLORATORY|nominal'),
 ("★M2 공시(1) 5% 기술 바닥 미달 ⟹ 헤드라인 승수 24/24 에 동반 의무 (본문 :4903-4905 축자 '헤드라인에 직접 붙는다')",
  r'승수.{0,8}가장높|highestwincount|24/24', r'기술바닥|5\\%기술|skillfloor|SPEC25가SS-REAL'),
 ("★M2 국소화 개입 종결 ⟹ -42.4% 급 크기 인용에 동반 의무 (본문 :4778-4780)",
  r'-42\.4|-42\.42', r'M2|국소화'),

]
for name, npat, apat in DUTY:
    rn, ra = re.compile(npat, re.I), re.compile(apat, re.I)
    print(f"\n== {name}")
    for i, f, b in FLAT:
        if not rn.search(f):
            continue
        lo, hi = max(1, i - 60), min(len(lines), i + 60)
        win = "".join(norm(x) for x in lines[lo-1:hi])
        ok = "✅" if ra.search(win) else "⛔없음"
        print(f"   {ok} :{i:<5} {b.strip()[:110]}")
