#!/usr/bin/env python3
"""논문 정합 감사 #3 — 인용금지 문구/수치의 기계 대조.

런 0회. reference_citation_bans.md 의 08-22~08-28 묶음에서 **손으로 옮기지 않고**
패턴을 정의하되, 각 패턴은 금지목록의 해당 줄을 근거로 함께 싣는다.
BAN_NUM 은 "본문에 이 수가 있으면 반드시 동반돼야 할 단서" 를 함께 검사한다.
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_bans.py
"""
import re, sys, hashlib, pathlib

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
 # ── 2026-08-28 (밤) 문헌 실사 묶음 (금지목록 "2026-08-28 (밤)" 절) ──
 ("⛔'순차 추적이 문헌의 틈' (08-28 §1-c)", r'유일한틈|순차추적이문헌의틈|축은열려있다'),
 ("⛔'Bergamasco 2021 = 자료 출처' (08-28 §2-a)", r'Bergamasco[^,\n]{0,20}자료(출처|집합)'),
 ("⛔'연속 파수 표현 자체가 신규' (08-28 §3-c)", r'연속파수[^,\n]{0,12}자체가신규|표현자체가신규'),
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
 # ⚠`게이트 24/24`(B-21·B-22 의 게이트 통과 수)는 SS-REAL 승수가 아니다 --- 등록 전
 #   실측에서 DUTY 13행 중 이 2행만이 오탐이었으므로 lookbehind 로 뺀다(2026-08-28).
 # ⚠**2026-08-28 (M-5b) 추가 제외**: `비트동일 24/24` 는 M-5b 음성대조(NOOP 팔이 기준 팔과
 #   obs_hash·nowcast 에서 비트 동일한 셀 수)이고 B-26 의 승수가 아니다. 검사기가 두 개의
 #   서로 다른 24/24 를 구별하지 못하는 것이 결함이었으므로 **두 번째 고정폭 lookbehind**
 #   로 뺀다(⛔논문 문안을 바꿔 검사기를 통과시키지 않았다 --- 그쪽이 더 나쁜 수리다).
 ("SS-REAL 승수 24/24 ⟹ F2(국소화 반경 고정) 공시 동반 의무",
  r'(?<!게이트)(?<!비트동일)24/24', r'국소화반경|loc_radius|loc/|Gaspari'),
 ("−42.42% ⟹ F2 공시 동반 의무", r'-42\.42', r'국소화반경|loc_radius|Gaspari'),
 ("'네 칸 유지/헤드라인이 산다' ⟹ 선형 참값 단서 동반 의무",
  r'네칸이모두유지|네칸이유지|allfourcellsholding', r'선형|linear'),
 ("−38.840%/48/48(B-20b) ⟹ blind 범위·계수기 공백 단서",
  r'-38\.84', r'blind|독립|사고|공백|계수기|replaces'),
 ("SS-LOWH 17/24 ⟹ EXPLORATORY/명목 병기", r'17/24', r'명목|탐색적|EXPLORATORY|nominal'),
 ("★M2 공시(1) 5% 기술 바닥 미달 ⟹ 헤드라인 승수 24/24 에 동반 의무 (본문 :4903-4905 축자 '헤드라인에 직접 붙는다')",
  r'승수.{0,8}가장높|highestwincount|(?<!게이트)(?<!비트동일)24/24', r'기술바닥|5\\%기술|skillfloor|SPEC25가SS-REAL'),
 ("★M2 국소화 개입 종결 ⟹ -42.4% 급 크기 인용에 동반 의무 (본문 :4778-4780)",
  r'-42\.4|-42\.42', r'M2|국소화'),
 ("Charnotskii 인용 ⟹ 열람 범위(초록) 단서 동반 의무 (08-28 §3-c)",
  r'Charnotskii', r'열람범위|초록'),
 ("선행 특허 인용 ⟹ 청구항 전문 미대조 단서 동반 의무 (08-28 §4)",
  r'10,?677,?891|Patent10677891', r'청구항전문|열람범위'),
 # ── 2026-08-28 추가: M-5b · M-4 · M-8b 의 결과-전 동반 의무 ────────────────
 # 근거: tmp/M5b_WINDOW_2026-08-28/RESULT.md §3·§7(밀도 공변 동반 의무, G-10 이 행 단위 강제) ·
 #       tmp/M4_CAPACITY_2026-08-28/RESULT.md §9 "병기 의무" ·
 #       tmp/M8b_LADDER_2026-08-28/RESULT.md §8 "⛔반드시 MDE 를 병기"
 ("★M-5b 창 크기(+31.3%) ⟹ 밀도 공변 공시 동반 의무",
  r'31\.3\\?%|31\.3', r'밀도|density|점/초'),
 ("★M-4 용량 크기(+1528%) ⟹ 사후 완화·원자 수준 공시 동반 의무",
  r'1528', r'완화|원자수준|원자의|사후'),
 ("★M-4 게이트 51/51 ⟹ 자기동결 표시 공시 동반 의무",
  r'51/51', r'동결뒤에|자기동결|완화|이동했을가능성'),
 ("★M-8b PRIMARY(파수 배치 고리) ⟹ MDE q=0.869 병기 의무",
  r'파수배치', r'0\.869'),
]
# ★(A) PHRASE 는 **자문**이다 --- 논문이 금지 자체를 축자로 인용하므로 정상 HIT 가 많다
#   (2026-08-28 실측: HIT 8종 전부가 "이렇게 쓰지 마라" 문장이거나 철회 표시 이력이었다).
#   ⟹ exit code 는 **(B) 단서 동반 의무**만 본다. 그쪽은 같은 날 오탐 2건을 제거해 0이다.
n_duty_fail = 0
for name, npat, apat in DUTY:
    rn, ra = re.compile(npat, re.I), re.compile(apat, re.I)
    print(f"\n== {name}")
    for i, f, b in FLAT:
        if not rn.search(f):
            continue
        lo, hi = max(1, i - 60), min(len(lines), i + 60)
        win = "".join(norm(x) for x in lines[lo-1:hi])
        hit = bool(ra.search(win))
        n_duty_fail += (not hit)
        print(f"   {'✅' if hit else '⛔없음'} :{i:<5} {b.strip()[:110]}")

print(f"\n단서 동반 의무 위반 {n_duty_fail}건 (PHRASE 절은 자문이므로 exit code 에 넣지 않는다)")
sys.exit(1 if n_duty_fail else 0)
