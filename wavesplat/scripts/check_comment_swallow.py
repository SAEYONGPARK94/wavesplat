#!/usr/bin/env python3
"""★**주석이 본문을 삼켰는지** 검사한다. ⛔2026-09-02 하루에 **일곱 번** 나온 결함이다.

기제
----
문자열 치환으로 주석 블록을 끼워 넣을 때, 매칭한 `old` 가 **줄 중간에서 끝나면**
그 줄의 남은 본문이 새 주석 블록의 **마지막 줄 꼬리**로 딸려 들어간다.
LaTeX 은 `%` 뒤를 전부 버리므로 **그 본문이 인쇄면에서 사라진다.**

⛔실제 피해(2026-09-02):
  · 학위논문 **국문 초록**에서 *"⚠단 합성 조건이며 …"* 공시가 사라지고
    *"기준선이 아니라 … 최강 기준선이다"* 라는 **자기모순 단편**만 인쇄됐다.
  · 학위논문 **영문 초록**에서 *"These are synthetic conditions …"* 가 사라졌다.
  · 논문 §3.4.1 에서 **체크포인트 SHA-256 문장 전체**가 사라졌다.
  · 그 밖 넷.
★한 건은 공시 검사기가 잡았지만(요구 자구가 사라져 FAIL), **나머지 여섯은
  아무 기계도 보지 못했다** --- 인쇄면이 문법만 이상해질 뿐 조판은 성공한다.

검사 방법 (정적, 런 0회)
------------------------
주석 블록 **바로 뒤**의 본문 줄이 **소문자 영문자로 시작**하는데, 주석 블록 **바로 앞**의
본문 줄이 **문장 끝 구두점**으로 끝났다면 = 「문장이 끝났는데 다음 문장이 소문자로
시작한다」 ⟹ 그 사이 주석이 문장 머리를 삼켰을 가능성이 높다.

⚠**한계**: 문장 중간에 일부러 주석을 넣으면 같은 모양이 된다. 그때는 주석을 문장 **위로**
옮겨라 --- 가드가 잡을 수 있게 쓰는 편이 가드를 느슨하게 하는 것보다 낫다(선례가
`paper/wavesplat.tex` 의 dedup 문단이다).
⚠한글 본문은 대소문자가 없어 이 검사가 **못 본다**. 그래서 `--diff` 모드가 따로 있다.

`--diff` 모드
-------------
`git diff` 를 읽어 **추가된 주석 줄**이 **삭제된 본문 줄**의 20자 이상 조각을 품고 있으면
경고한다. 언어와 무관하게 「본문이 주석으로 옮겨갔다」를 직접 잡는다.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS = ('paper/wavesplat.tex', 'paper/changes_appendix.tex', 'thesis/thesis_main.tex')
SENT_END = re.compile(r'[.!?]["\')\}]?\s*$')
# ⚠신호 2 — **한글 본문용**. 신호 1(주석 뒤 소문자 시작)은 대소문자가 있는 언어만 본다.
#   학위논문 초록 두 건이 그래서 신호 1 에 안 걸렸다. ⟹ **주석 줄 자신**을 본다:
#   마지막 문장 끝 **뒤에** LaTeX 본문 매크로가 나오면 그 꼬리는 본문일 공산이 크다
#   (주석은 매크로를 인용하더라도 보통 문장 안에서 인용한다).
SENT_SPLIT = re.compile(r'(?:[.!?]|다\.|⟹)\s+')
BODY_MACRO = re.compile(r'\\(?:textbf|emph|texttt|ref|cite|label|item)\{')


def static(paths):
    bad = []
    for rel in paths:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        prev, incomment = None, False
        for i, line in enumerate(open(p, errors='replace').read().split('\n')):
            st = line.lstrip()
            if st.startswith('%'):
                incomment = True
                m = list(SENT_SPLIT.finditer(st))
                # ⚠**첫** 문장 경계 이후를 본다 --- 마지막 경계만 보면 삼킨 본문이 스스로
                #   문장을 끝낼 때 놓친다(자가시험 `SHA문장-삼킴` 이 그렇게 잡혔다).
                if m and BODY_MACRO.search(st[m[0].end():]):
                    bad.append((rel, i + 1,
                                '주석 꼬리에 본문 매크로 → ' + st[m[0].end():][:60]))
                continue
            if not st:
                incomment, prev = False, None
                continue
            if incomment and prev is not None and SENT_END.search(prev) \
                    and re.match(r'[a-z]', st):
                bad.append((rel, i + 1, '주석 뒤 본문이 소문자로 시작 → ' + st[:60]))
            incomment, prev = False, line
    return bad


def diffmode(rev):
    """추가된 주석 줄이 삭제된 본문 줄의 조각을 품는가."""
    cmd = ['git', 'diff', '-U0', rev, '--', '*.tex']
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    removed, bad = [], []
    for line in out.split('\n'):
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('-') and not line[1:].lstrip().startswith('%'):
            removed.append(line[1:].strip())
        elif line.startswith('+') and line[1:].lstrip().startswith('%'):
            add = line[1:]
            for r in removed:
                # 삭제된 본문 줄의 **꼬리** 20자 이상이 주석 줄 안에 있는가
                for n in range(len(r), 19, -1):
                    if r[-n:] in add:
                        bad.append((add.strip()[:60], r[-n:][:60]))
                        break
                else:
                    continue
                break
    return bad


FORGERIES = [
    # (이름, 위조 줄, 기대 신호)
    ('한글초록-삼킴',
     '%   ⚠12 중 4 는 앞선 배치가 이미 썼다. ⚠단 \\textbf{합성 조건}이며, ``강화된\'\'',
     'comment'),
    ('영문초록-삼킴',
     '%   최대 시드 = 1,703. ⚠These are \\textbf{synthetic} conditions, and the',
     'comment'),
    ('SHA문장-삼킴',
     "%   보장되지 않는다. The checkpoint's SHA-256 is \\texttt{\\CkptShaFull{}}. We are",
     'comment'),
]


def selftest():
    """★위조가 반드시 잡혀야 한다(규율 A) --- 그리고 정상 주석은 조용해야 한다(B)."""
    ok = True
    print('── 위조 시험 ' + '─' * 40)
    for name, line, _ in FORGERIES:
        m = list(SENT_SPLIT.finditer(line.lstrip()))
        hit = bool(m and BODY_MACRO.search(line.lstrip()[m[0].end():]))
        print(('  ✅' if hit else '  ⛔') + f' {name}: ' + ('잡힘' if hit else '**놓침**'))
        ok &= hit
    benign = [
        '% ⛔[2026-09-02] 앞 판은 *"the gap is small"* 이라 적었다. 이웃 평가가 걸쳐 있다.',
        '%   `\\textbf{합성}` 을 문장 안에서 인용한다 --- 이것은 잡히면 안 된다.',
        '% ★재생성이 곧 검사다.',
    ]
    for b in benign:
        m = list(SENT_SPLIT.finditer(b.lstrip()))
        hit = bool(m and BODY_MACRO.search(b.lstrip()[m[0].end():]))
        print(('  ✅' if not hit else '  ⛔') + f' 정상주석: ' + ('침묵' if not hit else '**오발동**'))
        ok &= not hit
    # 신호 1
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False, dir=ROOT) as f:
        f.write('Body sentence ends here.\n% a comment block\ncontinues lowercase.\n')
        tmp = os.path.relpath(f.name, ROOT)
    hit = bool(static([tmp]))
    os.unlink(os.path.join(ROOT, tmp))
    print(('  ✅' if hit else '  ⛔') + ' 소문자시작-위조: ' + ('잡힘' if hit else '**놓침**'))
    ok &= hit
    print('  ' + ('✅자가시험 통과' if ok else '⛔자가시험 실패'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--diff', metavar='REV',
                    help='이 리비전과의 diff 에서 「본문 → 주석」 이동을 찾는다')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print('── 주석 삼킴 검사 ' + '─' * 36)
    bad = static(DOCS)
    for rel, ln, txt in bad:
        print(f'  ⛔{rel}:{ln}  주석 뒤 본문이 소문자로 시작한다 → {txt}')
    if a.diff:
        for add, frag in diffmode(a.diff):
            print(f'  ⛔본문이 주석으로 들어갔다: "{frag}" ⊂ "{add}"')
            bad.append((a.diff, 0, frag))
    if bad:
        print(f'  [FAIL] 후보 {len(bad)}건 — 주석이 본문을 삼켰는지 그 줄을 읽어라.')
        return 1
    print(f'  [OK  ] {len(DOCS)} 문서 · 삼킴 후보 0건')
    return 0


if __name__ == '__main__':
    sys.exit(main())
