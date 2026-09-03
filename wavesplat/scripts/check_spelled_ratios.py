#!/usr/bin/env python3
"""★**철자로 쓴 비율**을 잡는다 — 매크로 방벽을 우회하는 경로 하나를 닫는다 (2026-09-03).

**왜 만들었나.** 오늘 적대적 감사가 `paper/waveprior.tex` 에서 이것을 찾았다:

    Our arm beats a per-record zero field on **eleven of twelve** throughout.

그 수(`\\ROneBShipFixed{}/\\ROneBN{}`)에는 `check_paper_disclosures.py` 의 규칙이 **셋**
걸려 있었다(다중성 · 클러스터 · 지지집합). 그런데 세 규칙 모두 **매크로 이름**을 트리거로
쓰므로, 같은 수를 **영어 철자**로 적는 순간 **세 규칙이 전부 침묵**했다.
⟹ 아크의 헤드라인 실측 수치가 **아무 공시 없이** 인쇄됐다.

★일반 규율: ***방벽이 매크로에 걸려 있으면 산문이 그 옆으로 걸어 나간다.***

**무엇을 잡나.** `<철자 숫자> of <철자 숫자>` 형태만 잡는다(예: *"eleven of twelve"*).
이 형태는 이 원고에서 **언제나 셈 비율**이고, 셈 비율은 생성기 매크로여야 한다.
⛔넓히지 않는다 --- 늑대를 외치면 무시당한다(저장소 규율).
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGETS = [os.path.join(ROOT, 'paper', 'waveprior.tex'),
           os.path.join(ROOT, 'thesis', 'thesis_main.tex')]

WORDS = ('one two three four five six seven eight nine ten eleven twelve thirteen '
         'fourteen fifteen sixteen seventeen eighteen nineteen twenty '
         'thirty forty fifty sixty').split()

# ⛔[2026-09-03 재검이 초판을 REFUTE 했다] 초판은 두 수가 **붙어 있을 때만** 잡았다.
#   실측: `"four of the seven"` · `"Only seven of the twelve records"` ·
#   `"three of the four relative magnitudes"` 가 **바로 이 두 파일에** 살아 있는데
#   검사기와 방벽 27개가 전부 초록이었다. `"out of"` 와 `"one"` 도 빠져 있었다.
#   ⟹ ★**「있는 것을 없다고」 하는 검사기는 없는 것보다 나쁘다** — 안전하다는 신호를 준다.
_W = '|'.join(WORDS)
_NUM = r'(?:' + _W + r'|\d{1,4})'
_DEN = r'(?:' + _W + r'|\d{1,4}|\\[A-Za-z]+\{\})'

# ⛔⛔[2026-09-03, 2차 감사] **초판과 1차 확장이 실문서에서 진짜 위반을 0/10 잡았다.**
#   초록은 전부 ALLOW 관용구가 만든 것이었고 **검출 실적이 0** 이었다.
#   ★그리고 `thesis_main.tex` 는 **한국어 문서인데 정규식이 영어 전용**이라
#   구조적으로 발동 불가였다 --- 저장소가 이름 붙인 「있는 것을 없다고 하는 검사기」다.
#   ⟹ 아래 셋을 추가한다: ①한국어 「N(개) 중 M(개)」 ②`none of` ③숫자·낱말 혼합과
#     매크로 분모. ⛔분수·서수·60 초과는 **넣지 않았다**(오탐이 늑대가 된다).
PATS = [
    # 영어: <수> of/out of [the] <수>   (분모에 매크로 허용)
    re.compile(r'\b(' + _NUM + r')\s+(?:out\s+of|of)\s+(?:the\s+)?(' + _DEN + r')\b',
               re.IGNORECASE),
    # 영어: none of [the] <수>
    re.compile(r'\b(none)\s+of\s+(?:the\s+)?(' + _DEN + r')\b', re.IGNORECASE),
    # ★한국어: 「N개 중 M개」 · 「N 중 M」 · 「N 가운데 M」
    re.compile(r'([\d,]{1,7})\s*(?:개|건|칸|레코드|블록|셀)?\s*(?:중|가운데)\s*'
               r'([\d,]{1,7})\s*(?:개|건|칸|레코드|블록|셀)?'),
]
PAT = PATS[0]          # 하위호환(자가시험이 참조한다)

# ⛔허용 목록 --- 셈 비율이 **아닌** 관용 표현만. 사유를 함께 적는다.
ALLOW = {
    # ⚠셈 비율이 **아닌** 선택 관용구. 각 항목에 사유를 적는다(2026-09-03).
    'one of the three': '선택 관용구 --- "any one of the three" = 셋 중 어느 하나에도 귀속 불가',
    'one of the two': '선택 관용구 --- "one of the two bands" = 두 대역 중 한쪽',
    'one of the same': '선택 관용구 --- "one of the same public records"',
    # ⚠[2026-09-03 2차] 셈 비율이지만 **결과 주장이 아닌** 예시·서지 통계.
    #   ⛔면제 사유를 반드시 적는다. 결과 수치가 여기 들어오면 그것은 규율 위반이다.
    '5개 중 1개': '가지치기 **예시**(BF3 5comp -> N_a=4) --- 판정 수치가 아니다',
    '217개 중 1': '저장소 커밋 서지 통계(1,217 중 1,203) --- 측정 결과가 아니다',
    '1,217개 중 1,203개': '같은 서지 통계 (구분자 포함 형태)',
}


def strip_comments(text):
    out = []
    for line in text.split('\n'):
        i, esc = None, False
        for k, ch in enumerate(line):
            if ch == '\\':
                esc = not esc
                continue
            if ch == '%' and not esc:
                i = k
                break
            esc = False
        out.append(line if i is None else line[:i])
    return '\n'.join(out)


def main():
    bad = []
    for p in TARGETS:
        if not os.path.exists(p):
            continue
        raw = open(p, encoding='utf-8').read()
        body = strip_comments(raw)
        seen = set()
        for pat in PATS:
            for m in pat.finditer(body):
                phrase = m.group(0)
                if phrase.lower().strip() in ALLOW:
                    continue
                line = body[:m.start()].count('\n') + 1
                if (line, phrase) in seen:
                    continue
                seen.add((line, phrase))
                bad.append((os.path.relpath(p, ROOT), line, phrase))
    print('── 철자 비율 검사 ' + '─' * 40)
    if bad:
        for f, l, ph in bad:
            print(f'  ⛔{f}:{l}  "{ph}"')
        print(f'\n⛔{len(bad)}건 --- 셈 비율은 **생성기 매크로**로 적어라.')
        print('   사유: 매크로에 걸린 공시 규칙이 철자 표기를 보지 못한다(2026-09-03 감사).')
        return 1
    print('  [OK  ] 철자로 쓴 셈 비율 0건')
    return 0


if __name__ == '__main__':
    sys.exit(main())
