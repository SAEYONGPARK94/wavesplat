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
     [r'synthetic', r'simulated'],
     'B-1 은 합성·선형 GT 단서 병기 의무'),
    ('B1-체크포인트', r'\\BOneRel|\\BOneBetter',
     [r'same distribution family', r'seed axis only', r'trained on the same'],
     'B-1 은 체크포인트가 학습 분포와 겹친다는 단서 병기 의무'),
    ('B1-최강아님', r'\\BOneRel|\\BOneBetter',
     [r'strongest .*we possess', r'strongest eligible', r'not the same claim as'],
     '"최강 적격" 이지 "강화된" 이 아니다'),
    ('B22-우변얇음', r'\\BTwentyTwoRel|\\BTwentyTwoBetter',
     [r'\\BTwentyTwoAnaZero', r'right-hand side of this contrast is'],
     'B-22 L2 의 우변(해석적)은 제로필드를 못 넘는다 — 병기 의무'),
    ('B16-교락폐쇄', r'\\BSixteenRel',
     [r'confound', r'not attributable by subtraction', r'refit it necessarily performs'],
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
    """(제목, 본문) 목록. 절 경계는 \\section / \\subsection."""
    parts, cur, title = [], [], '<preamble>'
    for line in text.split('\n'):
        if re.match(r'\\(sub)?section\{', line):
            parts.append((title, '\n'.join(cur)))
            title, cur = line.strip(), []
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


def main():
    if not os.path.exists(TEX):
        raise SystemExit(f'없다: {TEX}')
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
    if fails:
        print(f'\n⛔공시 누락 {len(fails)}건:')
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
