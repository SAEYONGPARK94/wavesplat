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


def sections(text):
    """(제목, 본문) 목록 --- ⛔**절 단위로** 본다.

    ⛔[2026-09-02 감사 M-7-1] 초판은 문서 전체를 한 문자열로 접었다. 그래서 한 곳의
    공시가 **다른 절의 무공시 인용을 면허**했다(근거를 전부 지워도 통과했다).
    ★논문판은 2026-09-01 에 정확히 이 결함을 고쳤는데 **이식되지 않았다** ---
    「같은 결함을 두 번 고치지 않으려면 처방도 두 문서에 옮겨야 한다」.
    """
    ABS = r'\\begin\{(abstract|englishabstract|onecolabstract)\}'
    parts, cur, title = [], [], '<frontmatter>'
    for line in text.split('\n'):
        env = re.match(ABS, line)
        if re.match(r'\\(sub)?section\*?\{|\\chapter\*?\{', line) or env:
            parts.append((title, '\n'.join(cur)))
            title = line.strip()[:60] if not env else '<' + env.group(1) + '>'
            cur = []
        else:
            cur.append(line)
    parts.append((title, '\n'.join(cur)))
    return parts


# (이름, 트리거 정규식, 요구 정규식 목록(하나라도 있으면 통과), 사유)
# ★[2026-09-03] 검사기가 실제로 놓쳤던 문자열. 트리거 회귀 시험용(⛔지우지 마라).
TRIGGER_REGRESSION = ('약 $39\\%$ 낮춘다($64$ 에피소드 중 $62$, 양측 $p = 2.3'
                      '\\times10^{-16}$; 배치 \\texttt{B-1}')

RULES = [
    # ★[2026-09-03 박사 기여 감사 C-3] 희소 용량--반응의 아래 계단은 **무기술**이다.
    #   `14/16` 만 인용하면 계단 크기를 오도한다 ⟹ 병기를 강제한다.
    ('BNineteen-아래계단',
     r'희소 층 \$14/16\$|\$14/16\$.{0,80}?B-19|B-19.{0,80}?\$14/16\$',
     [r'KGRID2', r'\$2/16\$', r'무기술'],
     'B-19 의 14/16 인용에는 KGRID2 의 제로필드 대비 2/16(무기술) 병기 의무'),
    # ⛔[2026-09-02 감사 M-7-3] 초판 트리거 `62/64` 는 **두 초록을 못 봤다**
    #   (국문 `$64$ 중 $62$` · 영문 `$62$ of $64$`) ⟹ 가장 눈에 띄는 인용이 검사 밖이었다.
    ('BOne-사후작성',
     # ⚠[2026-09-02] `62/64` 단독은 **다른 양**에도 걸린다(W-15 의 비교자 대 제로).
     #   ⟹ **`B-1` 태그와의 근접**을 요구한다.
     # ⛔[2026-09-03] 이 정규식이 **실제 결함을 놓쳤다**: 기여 목록(`:1647`)의 문자열은
     #   `$64$ **에피소드** 중 $62$` 인데 위 패턴은 `$64$ 중 $62$` 만 잡았다 ⟹ 검사기가
     #   *"누락 0"* 을 찍는 동안 논문에서 가장 많이 인용될 수에 사후작성 공시가 없었다.
     #   ★같은 실패 모드(「있는 것을 없다고」의 역방향 = 「없는 것을 있다고」)의 재발.
     #   ⟹ 두 수 사이에 **한 낱말까지 허용**한다.
     r'(?:62/64|\$64\$(?:\s*\S{1,6})?\s*중\s*\$62\$|\$62\$ of \$64\$).{0,160}?B-1'
     r'|B-1.{0,160}?(?:62/64|\$64\$(?:\s*\S{1,6})?\s*중\s*\$62\$|\$62\$ of \$64\$)',
     # ⚠검사기는 **원문**을 본다 --- 매크로는 조판 시점에야 문장이 되므로
     #   매크로 이름 자체를 요구 목록에 넣어야 한다(안 그러면 참인 공시를 누락으로 오판한다).
     [r'\\BOnePostHoc(?:EN)?\{\}', r'게이트\$?\\cdot\$?분석 코드는 결과 뒤에 작성',
      r'결과 뒤에 작성되었다', r'written after (?:its |the )?results'],
     'B-1 의 게이트·분석은 결과 뒤에 작성됐다(verdict_b1.json `_honesty`) — 병기 의무'),
    # ⛔[2026-09-02 감사 M-7-2] 초판은 **항진**이었다 --- 트리거가 `\newcommand` 정의
    #   자신에 매치되므로 사용처를 전부 지워도 발동·통과한다.
    #   ⟹ 트리거를 **크기 자구**로 바꾸고 요구를 **매크로 사용**으로 바꾼다.
    # ⚠[2026-09-02] 초판 트리거는 **금지 문장**에도 걸렸다(*"약 $-40\%$ 를 약 $-39\%$ 와
    #   직접 비교하지 않는다"* --- 그것은 크기 주장이 아니라 그 반대다).
    #   ⟹ 크기가 **주장으로 서는 형태**(승수 병기 또는 "낮춘다"/"lowers")만 트리거한다.
    ('BOne-비교자강도',
     r'약 \$39\\%\$[^。]{0,80}?(?:낮춘다|\$62\$)|by about \$39\\%\$[^.]{0,80}?relative to',
     [r'\\CmpSkillBOne(?:EN)?\{\}'],
     'B-1 크기를 적는 자리에는 그 코퍼스의 비교자 강도를 병기한다'),
    # ⚠[2026-09-02] 의무는 **매크로**가 아니라 **공시**다 --- §결과 절은 표 칸에
    #   `SS-LOWH … 명목` 으로 이미 공시한다. 매크로만 요구하면 참인 공시를 누락으로 오판한다.
    ('SSLowh-명목',
     r'네 칸이 \\textbf\{모두 유지\}|all four cells holding|네 칸이 모두 유지',
     [r'\\SSLowhNominal', r'SS-LOWH\}?[^\n]{0,120}?명목', r'명목[^\n]{0,80}?SS-LOWH',
      r'탐색적'],
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
    fired, missing = [], []
    for title, chunk in sections(text):
        b = body(chunk)
        for name, trig, need, why in RULES:
            if not re.search(trig, b):
                continue
            fired.append((name, title))
            if not any(re.search(x, b) for x in need):
                missing.append((name, title, why))
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
    for name, title, why in missing:
        print(f'  ⛔{name}  [{title}]: {why}')
    for m in pin:
        print(f'  {m}')
    if missing or pin:
        print(f'  [FAIL] 누락 {len(missing)} · 값 불일치 {len(pin)}')
        return 1
    print(f'  [OK  ] {len(RULES)} 규칙 중 {len(fired)} 발동 · 누락 0 · 값 핀 일치')
    return 0


if __name__ == '__main__':
    sys.exit(main())
