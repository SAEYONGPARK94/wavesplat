#!/usr/bin/env python3
"""★**동반 공시 의무**를 기계로 검사한다 — `gen_paper_constants.py` 의 사각지대.

배경(2026-09-01): 논문 개정에 적대적 감사를 붙였더니 `paper_constants.tex` 의 **수치 핀은
34원자 전건 통과**했는데 **BLOCKER 5건 중 넷과 MAJOR 대부분이 「공시 누락」**이었다.
감사 축자:

> *"생성기는 **수치**만 핀하고 **병기 의무**는 전혀 검사하지 않는다 --- BLOCKER 4·MAJOR 3 이
> 전부 그 사각지대다."*

⟹ 수치가 옳아도 **그 옆에 있어야 할 문장이 없으면** 그 인용은 금지목록 위반이다.
이 스크립트가 그것을 검사한다.

규약
----
`RULES` 의 각 항목은 *"이 매크로(또는 문자열)가 본문에 나타나면, 이 근거 표현들이
**같은 절 안에** 있어야 한다"* 를 뜻한다. 근거 표현은 **여러 대안** 중 하나면 된다
(문면은 바뀔 수 있으므로 의미 단서를 여러 개 둔다).

⛔이 검사기가 통과한다고 공시가 **충분**하다는 뜻은 아니다 --- **누락**만 잡는다.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TEX = os.path.join(ROOT, 'paper', 'wavesplat.tex')

# (이름, 발동 정규식, [대안 근거 정규식 …], 사유)
RULES = [
    ('T4-비교자붕괴', r'\\TFourRel|\\TFourBetter',
     [r'comparator (?:degrades|collapse)', r'\\TFourCmp\b', r'only \\TFourCmp'],
     'T-4 크기는 비교자 붕괴(47/48 -> 34/48) 병기 없이 인용 금지'),
    ('T4-rung범위', r'\\TFourRel|\\TFourBetter',
     [r'\$11\$ of (?:the )?\$?12', r'11 of the 12', r'lower\b.*\bend of the measured lead'],
     'T-4 rung 은 실측 밴드의 아래쪽 — "실측과 같은 지평" 금지'),
    ('B1-합성단서', r'\\BOneRel|\\BOneBetter',
     # 2차 감사: 'synthetic' 한 토큰이면 통과 = 사실상 항진이었다. 문맥을 요구한다.
     [r'synthetic\)', r'are synthetic', r'gains are synthetic', r'synthetic (?:corpus|seas|evaluation)'],
     'B-1 은 합성·선형 GT 단서 병기 의무'),
    ('B1-체크포인트', r'\\BOneRel|\\BOneBetter',
     [r'same distribution family', r'seed axis only', r'trained on the same'],
     'B-1 은 체크포인트가 학습 분포와 겹친다는 단서 병기 의무'),
    ('B1-최강아님', r'\\BOneRel|\\BOneBetter',
     [r'strongest .*we possess', r'strongest eligible', r'not the same claim as'],
     '"최강 적격" 이지 "강화된" 이 아니다'),
    ('B22-우변얇음', r'\\BTwentyTwoRel|\\BTwentyTwoBetter',
     [r'\\BTwentyTwoAnaZero', r'right-hand side of this contrast is',
      r'comparators are themselves weak'],
     'B-22 L2 의 우변(해석적)은 제로필드를 못 넘는다 — 병기 의무'),
    ('B16-교락폐쇄', r'\\BSixteenRel',
     # 2차 감사: 'confound' 한 토큰이면 통과했다(부정문도 통과). 확인 서술을 요구한다.
     [r'not confounded with', r'checked .{0,40}for confounding', r'not attributable by subtraction'],
     'B-16 크기 귀속은 B-21(교락 무시가능)이 지탱한다 — 병기 의무'),
    ('절제크기-결합금지', r'\\BSixteenRel',
     [r'not additive', r'should not add or subtract', r'no defined meaning'],
     'B-1/B-16/B-22 크기 직접비교·산술결합 금지'),
    # ⚠발동을 **우리 음성 주장**에만 건다(관련연구의 "measured records" 는 대상이 아니다).
    # 발동을 **12 레코드 음성을 서술하는 문장**에만 건다.
    ('실측음성-자구', r'(?:learned prior|representation) (?:is |does )[^.]{0,120}(?:zero-field|measured record)',
     [r'decision threshold', r'pre-registered .*threshold'],
     'T-5: "without skill" 금지 -> "유의/결정 문턱에 도달하지 못한다"'),
    # ⚠발동을 **절차를 총괄 서술하는 문장**에만 건다.
    ('사전등록-전칭금지', r'comparisons .{0,40}pre-registered',
     [r'written after the runs', r'not blind', r'two limits of that procedure'],
     '"All comparisons were pre-registered" 전칭 금지 — B-1 분석코드는 사후 작성'),
    ('MTen-blind', r'\\MTenRel|\\MTenBetter',
     [r'[Nn]ot blind'],
     'M-10 은 blind 아님 — 병기 의무'),
]


def sections(text):
    """(제목, 본문) 목록.

    절 경계는 `\\section` / `\\subsection`, **그리고 `abstract`·`highlights` 환경**이다.
    ⛔초판은 첫 `\\section` 이전을 한 덩어리(`<preamble>`)로 묶었고, 그래서
    **초록의 공시가 하이라이트의 무공시 인용을 면허**했다(2차 감사 결함 i:
    "highlights 에 비교자붕괴 0인데 rc=0").
    """
    parts, cur, title = [], [], '<frontmatter>'
    for line in text.split('\n'):
        env = re.match(r'\\begin\{(abstract|highlights)\}', line)
        endenv = re.match(r'\\end\{(abstract|highlights)\}', line)
        if re.match(r'\\(sub)?section\{', line) or env:
            parts.append((title, '\n'.join(cur)))
            title = line.strip() if not env else f'<{env.group(1)}>'
            cur = []
        elif endenv:
            parts.append((title, '\n'.join(cur)))
            title, cur = '<frontmatter>', []
        else:
            cur.append(line)
    parts.append((title, '\n'.join(cur)))
    return parts


def strip_comments(t):
    """주석을 빼고 **공백을 정규화**한다.

    ⛔초판은 줄바꿈을 그대로 뒀고, 그래서 "the comparator\ndegrades" 같은 정상 문장을
    **공시 누락으로 오판**했다(첫 실행에서 9건 중 여럿이 그것이었다).
    ★검사기가 「있는 것을 없다고」 하는 것도 「없는 것을 있다고」 하는 것만큼 나쁘다.
    """
    out = []
    for line in t.split('\n'):
        if line.lstrip().startswith('%'):
            continue
        out.append(re.sub(r'(?<!\\)%.*$', '', line))
    return re.sub(r'\s+', ' ', ' '.join(out))


def literals_outside_macros():
    """★**매크로 값이 tex 에 리터럴로 나타나는가.**

    3차 감사(2026-09-01)의 최대 구조적 지적:

    > *"매크로를 리터럴 수치로 치환 + 공시 문장 삭제 → 검사기가 `발동 없음` ×10 ·
    > rc=0 · '공시 확인' 을 찍는다. 이것은 가설이 아니다 --- 팔간 주장의 수치가
    > **이미 매크로 밖 리터럴**이라 BLOCKER 전건이 검사기 사각지대에 있었다."*

    ⟹ 공시 규칙은 **매크로 사용**에 걸려 있으므로, 누군가 값을 손으로 적으면 규칙이
    **조용히 침묵**한다. 이 검사가 그 우회를 막는다: `paper_constants.tex` 가 정의하는
    값이 tex 본문에 **그대로** 나타나면 실패한다(그 자리는 매크로를 써야 한다).
    """
    pc = os.path.join(ROOT, 'paper', 'paper_constants.tex')
    if not os.path.exists(pc):
        return True, '상수 파일 없음 — 건너뜀'
    vals = {}
    for line in open(pc, encoding='utf-8'):
        m = re.match(r'\\newcommand\{\\(\w+)\}\{(.+?)\}\s*(?:%|$)', line)
        if m:
            v = m.group(2).strip()
            # ⚠**오탐을 내면 안 된다.** 첫 판은 `10^{-4}`(평활 계수 lambda_s)와
            #   `0.005`(ETKF 관측잡음 sigma)를 잡았다 --- 매크로 값과 **우연히 같은
            #   문자열**인 다른 양이다. ★검사기가 늑대를 외치면 사람이 무시하기 시작한다.
            #   ⟹ **결과 크기**(퍼센트)만 본다: 그것이 공시 의무를 지는 값이고,
            #   손으로 적히면 규칙이 침묵하는 바로 그 값이다.
            if '\\%' in v:
                vals[m.group(1)] = v
    body = strip_comments(open(TEX, encoding='utf-8').read())
    hits = [(k, v) for k, v in vals.items() if v in body]
    if hits:
        return False, f'매크로 값이 리터럴로 인쇄됨 {len(hits)}건: ' + \
            ', '.join(f'\\{k}={v}' for k, v in hits[:3])
    return True, f'{len(vals)} 값 검사 · 리터럴 0'


def constants_up_to_date():
    """★커밋된 `paper_constants.tex` 가 생성물과 같은가.

    2차 감사: *"생성기를 안 돌리고 tex 만 손으로 고치면 무탐지"* --- CI·훅이 0개다.
    ⟹ 공시 검사기가 그 확인도 함께 한다(한 명령으로 둘 다).
    """
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, 'gen_paper_constants.py'),
                        '--check'], capture_output=True, text=True)
    tail = (r.stdout.strip().split('\n') or [''])[-1]
    return r.returncode == 0, tail


def main():
    if not os.path.exists(TEX):
        raise SystemExit(f'없다: {TEX}')
    ok_const, msg = constants_up_to_date()
    print(f'  [{"OK  " if ok_const else "FAIL"}] {"상수파일-최신":22s} {msg}')
    ok_lit, msg_lit = literals_outside_macros()
    print(f'  [{"OK  " if ok_lit else "FAIL"}] {"리터럴-우회":22s} {msg_lit}')
    raw = open(TEX, encoding='utf-8').read()
    fails = []
    print('── 동반 공시 검사 ─────────────────────────────────────────────')
    for name, trig, needs, why in RULES:
        hit_sections, ok_sections = [], []
        for title, body in sections(raw):
            b = strip_comments(body)
            if not re.search(trig, b):
                continue
            hit_sections.append(title[:52])
            if any(re.search(p, b) for p in needs):
                ok_sections.append(title[:52])
        if not hit_sections:
            print(f'  [--  ] {name:22s} (발동 없음)')
            continue
        missing = [t for t in hit_sections if t not in ok_sections]
        if missing:
            fails.append((name, why, missing))
            print(f'  [FAIL] {name:22s} 공시 누락 절 {len(missing)}: {missing[:2]}')
        else:
            print(f'  [OK  ] {name:22s} 발동 {len(hit_sections)}절 전건 공시')
    print('──────────────────────────────────────────────────────────────')
    if not ok_const:
        fails.append(('상수파일-최신', 'paper_constants.tex 가 생성물과 다르다', ['<repo>']))
    if not ok_lit:
        fails.append(('리터럴-우회', '매크로 값을 손으로 적으면 공시 규칙이 침묵한다',
                      ['<paper/wavesplat.tex>']))
    if fails:
        print(f'\n⛔실패 {len(fails)}건:')
        for n, why, ms in fails:
            print(f'  · {n}: {why}')
            for m in ms:
                print(f'      절: {m}')
        print('\n⚠수치가 옳아도 병기가 없으면 그 인용은 금지목록 위반이다.')
        return 1
    print('✅발동한 모든 절에서 동반 공시 확인')
    return 0


if __name__ == '__main__':
    sys.exit(main())
