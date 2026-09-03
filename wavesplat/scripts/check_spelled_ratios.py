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

WORDS = ('two three four five six seven eight nine ten eleven twelve thirteen '
         'fourteen fifteen sixteen seventeen eighteen nineteen twenty '
         'thirty forty fifty sixty').split()
PAT = re.compile(r'\b(' + '|'.join(WORDS) + r')\s+of\s+(' + '|'.join(WORDS) + r')\b',
                 re.IGNORECASE)

# ⛔허용 목록 --- 셈 비율이 **아닌** 관용 표현만. 사유를 함께 적는다.
ALLOW = {
    # (없음) --- 생기면 여기에 사유와 함께 추가한다.
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
        for m in PAT.finditer(body):
            phrase = m.group(0)
            if phrase.lower() in ALLOW:
                continue
            line = body[:m.start()].count('\n') + 1
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
