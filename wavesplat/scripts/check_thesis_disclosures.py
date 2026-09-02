#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★**학위논문 동반 공시 검사기** --- `check_paper_disclosures.py` 의 학위논문판.

⛔왜 이것이 필요한가 (2026-09-02 적대적 감사 A6·A10·A11 이 **셋 다 같은 원인**을 지목했다):

> `scripts/check_paper_disclosures.py:28` 의 대상은 `paper/waveprior.tex` **하나**다.
> ⟹ 학위논문의 의무 공시는 **아무도 기계로 보지 않는다.**

그래서 오늘 감사가 셋을 찾았다 --- `B-1` 의 *"게이트·분석이 결과 뒤에 작성됐다"*(0회) ·
`B-26` `SS-LOWH` 의 *"명목/탐색적"*(다섯 곳 중 0곳) · `M-10` 의 *"blind 아님"*(0회).
셋 다 **동결 판정 파일이 결과 시점에 직접 요구한** 공시다.

⚠**범위를 정직하게 적는다**: 이 검사기는 논문판의 35규칙을 이식한 것이 **아니다**.
오늘 수리한 **네 건**의 공시가 다시 사라지는 것을 막는다. 규칙은 늘려도 좋으나,
늘릴 때마다 `--selftest` 의 A/B/C 를 통과시켜라.

★그리고 ⛔**요구하는 자구가 참인지**(실패 모드 D)를 값으로 붙든다: `A9` 규칙은
비교자 강도 두 수를 **동결 행에서 다시 계산**해 문서의 값과 대조한다.
"""
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEX = os.path.join(ROOT, 'thesis', 'thesis_main.tex')


def body(text):
    """주석 제거 + 공백 정규화. ⛔주석은 이 자구를 **인용해서** 설명하므로 세지 않는다."""
    out = []
    for line in text.split('\n'):
        if line.lstrip().startswith('%'):
            continue
        out.append(re.sub(r'(?<!\\)%.*$', '', line))
    return re.sub(r'\s+', ' ', ' '.join(out))


# (이름, 트리거 정규식, 요구 정규식 목록(하나라도 있으면 통과), 사유)
RULES = [
    ('BOne-사후작성',
     r'62/64',
     [r'게이트\$?\\cdot\$?분석 코드는 결과 뒤에 작성', r'결과 뒤에 작성되었다'],
     'B-1 의 게이트·분석은 결과 뒤에 작성됐다(verdict_b1.json `_honesty`) — 병기 의무'),
    ('BOne-비교자강도',
     r'\\CmpSkillBOne',
     [r'\\newcommand\{\\CmpSkillBOne\}'],
     'B-1 크기를 적는 자리에는 그 코퍼스의 비교자 강도를 병기한다'),
    ('SSLowh-명목',
     r'네 칸이 \\textbf\{모두 유지\}|all four cells holding|네 칸이 모두 유지',
     [r'\\SSLowhNominal'],
     'verdict_b26.json corners.SS-LOWH.reading 이 「가족단위 미달 = 명목」 병기를 요구한다'),
    ('MTen-blind',
     r'동반해야 하는 공시',
     [r'blind 가 아니다'],
     'verdict_m10.json disclosures[0] = 「blind 아님 + 스모크 노출」 — 병기 의무'),
]


def value_pin():
    """★실패 모드 D --- 요구하는 자구의 **수가 참인가**를 동결 행에서 다시 계산한다."""
    p = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14', 'results.jsonl')
    if not os.path.exists(p):
        return []
    byep = defaultdict(list)
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('arm') != 'SPEC25' or 'error' in r:
            continue
        m = re.search(r'(\d+)comp', r['episode'])
        if m and int(m.group(1)) in (15, 30):
            byep[r['episode']].append(r)
    if not byep:
        return []
    win = n = 0
    rels = []
    for v in byep.values():
        a = st.median([x['avg_nRMSE'] for x in v])
        z = st.median([x['zero_nRMSE'] for x in v])
        n += 1
        win += a < z
        rels.append((a - z) / z * 100)
    med = st.median(rels)
    txt = open(TEX, encoding='utf-8').read()
    # ⛔**매크로 정의 안에서만** 찾는다. `$64/64$` 는 다른 줄(무기술 대비 열)에도 있어서
    #    문서 전체를 뒤지면 매크로의 수를 바꿔도 통과한다 = 핀이 공허해진다.
    m = re.search(r'\\newcommand\{\\CmpSkillBOne\}\{(.{0,400}?)\}\n', txt, re.S)
    if not m:
        return ['⛔`\\CmpSkillBOne` 매크로 정의를 찾지 못했다']
    mac, bad = m.group(1), []
    if f'${win}/{n}$' not in mac:
        bad.append(f'⛔`\\CmpSkillBOne` 의 승수가 재계산값 {win}/{n} 과 다르다')
    if f'{med:.2f}' not in mac:
        bad.append(f'⛔`\\CmpSkillBOne` 의 상대 중앙이 재계산값 {med:.2f}% 와 다르다')
    return bad


def run(text):
    b = body(text)
    fired, missing = [], []
    for name, trig, need, why in RULES:
        if not re.search(trig, b):
            continue
        fired.append(name)
        if not any(re.search(x, b) for x in need):
            missing.append((name, why))
    return fired, missing


def selftest():
    """A 위조 · B 침묵 · C 미검증 --- `check_paper_disclosures.py --selftest` 와 같은 규약."""
    text = open(TEX, encoding='utf-8').read()
    b = body(text)
    a_ok = b_ok = 0
    unfired = []
    for name, trig, need, _ in RULES:
        if not re.search(trig, b):
            unfired.append(name)
            continue
        forged = b
        for x in need:
            forged = re.sub(x, 'XXFORGEDXX', forged)
        if any(re.search(x, forged) for x in need):
            print(f'  ⛔A 실패 {name}: 근거를 지웠는데도 통과한다')
        else:
            a_ok += 1
        silent = re.sub(trig, 'XXNOTRIGXX', b)
        if re.search(trig, silent):
            print(f'  ⛔B 실패 {name}: 트리거를 지웠는데 여전히 발동한다')
        else:
            b_ok += 1
    print(f'── 학위논문 공시 항진 검사 {"─" * 26}')
    print(f'  A 위조 {a_ok}/{len(RULES) - len(unfired)} · '
          f'B 침묵 {b_ok}/{len(RULES) - len(unfired)} · C 미검증 {len(unfired)}')
    if unfired:
        print('  ⚠발동하지 않은 규칙(=미검증, 「통과」로 세지 않는다): ' + ', '.join(unfired))
    ok = (a_ok == b_ok == len(RULES) - len(unfired)) and not unfired
    print('  ' + ('[OK  ] 전부 통과' if ok else '[FAIL] 위 항목을 읽어라'))
    return 0 if ok else 1


def main():
    if '--selftest' in sys.argv:
        return selftest()
    text = open(TEX, encoding='utf-8').read()
    fired, missing = run(text)
    pin = value_pin()
    print('── 학위논문 동반 공시 ' + '─' * 32)
    for name, why in missing:
        print(f'  ⛔{name}: {why}')
    for m in pin:
        print(f'  {m}')
    if missing or pin:
        print(f'  [FAIL] 누락 {len(missing)} · 값 불일치 {len(pin)}')
        return 1
    print(f'  [OK  ] {len(RULES)} 규칙 중 {len(fired)} 발동 · 누락 0 · 값 핀 일치')
    return 0


if __name__ == '__main__':
    sys.exit(main())
