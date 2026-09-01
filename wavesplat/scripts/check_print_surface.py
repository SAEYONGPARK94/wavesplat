#!/usr/bin/env python3
"""★**인쇄면 검사기** --- 조판된 PDF 의 문장 자체가 망가졌는가.

⛔왜(2026-09-02, 검증 감사의 마지막 지적, 축자):

> *"All 13 barriers are green while the paper carries a **predicateless sentence**, a
>  **duplicated heading**, **eight stuttering references**, an **ungrammatical clause**,
>  and a §5.2/§5.7 contradiction. The suite checks numbers, disclosures, bans and
>  freezes … and there is **no print-surface check**."*

맞는 지적이다. 오늘 하루에만 내 수리가 인쇄를 **다섯 번** 깼다(맹목 치환 · 술어 삭제 ·
중복 제목 · 말더듬 참조 · 고아 문장). 그 다섯은 전부 **grep 한 줄**로 잡힌다.

⚠**한계**: 문법을 판정하지 않는다. **알려진 손상 패턴**만 본다.
   새 패턴은 사람이 인쇄면을 읽어야 찾는다 --- 이 검사기는 그것을 대신하지 않는다.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PDFS = [('paper', 'paper/wavesplat.pdf'), ('thesis', 'thesis/thesis_main.pdf')]

# (이름, 정규식, 무엇이 잘못인가)
PATTERNS = [
    ('말더듬 참조', r'\b(App|Fig|Sec|Tab)\.\s+(Appendix|Figure|Section|Table)\b',
     '`App.~\\ref{}` 처럼 접두어와 라벨이 겹쳐 인쇄된다'),
    ('빈 참조', r'\?\?', '미해결 참조가 `??` 로 인쇄된다'),
    # ⚠초판은 목차.표목차의 **점선 리더**(`. . . . .`)를 전부 잡았다(오탐 다수).
    #   ⟹ 뒤에 리더가 이어지지 않는 경우만 본다.
    ('이중 마침표', r'[a-z가-힣]\.\s\.\s(?!\.)', '문장 끝이 두 번 찍힌다(치환 사고)'),
    # ⚠코드 리스팅의 무인자 호출(`W.pop_oldest()`)이 잡힌다 ⟹ 공백이 실제로 있을 때만.
    ('빈 괄호', r'\(\s+\)', '괄호 안이 비었다(내용이 지워졌다)'),

    # ⚠LaTeX 의 `'` 는 **U+2019** 로 조판된다 --- ASCII 만 보면 못 잡는다(항진 검사가 잡았다).
    ('스트레이 어포스트로피', r"\b[a-z]+s['\u2019],\s", "복수형 뒤 소유격 쉼표(비문 징후)"),
    # ⚠정상 문장 *"the tuned configuration is (Ne=100, …)"* 가 잡힌다
    #   ⟹ 괄호 안이 **참조**(App./Fig./Sec./Table)일 때만 = 술어가 지워진 징후.
    ('is 다음 참조괄호', r'\bis\s*\((?:App|Fig|Sec|Table|Appendix|Figure|Section)\b',
     "술어가 지워지고 참조 괄호만 남았다(오늘 실제로 났다)"),
]


def text_of(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    env = dict(os.environ)
    env['PATH'] = os.path.expanduser('~/texlive/2026/bin/x86_64-linux') + ':' + env['PATH']
    r = subprocess.run(['pdftotext', '-layout', p, '-'], capture_output=True,
                       text=True, env=env)
    return r.stdout if r.returncode == 0 else None


def orphan_fragments(txt):
    """★진짜 고아 단편 = **앞뒤가 빈 줄인** 짧은 소문자 줄.

    ⛔초판은 *"소문자로 시작해 마침표로 끝나는 줄"* 을 잡았다가 **98건 오탐**을 냈다 ---
    그것은 **줄바꿈된 문단의 마지막 줄** 전부다. 오늘 세 번째로 같은 실수를 했다.
    ⟹ 문단 구조를 쓴다: 위아래가 비어 있어야 고아다.
    """
    lines = [re.sub(r'^\s*\d+\s*', '', l).rstrip() for l in txt.split('\n')]
    out = []
    # ⚠표 행은 같은 값이 반복될 수 있다 ⟹ 숫자.기호만으로 이뤄진 줄은 제외한다.
    for i in range(1, len(lines) - 1):
        t = lines[i].strip()
        if not t or len(t) > 42:
            continue
        if not re.match(r'^[a-z]', t) or not t.endswith('.'):
            continue
        if lines[i - 1].strip() or lines[i + 1].strip():
            continue
        out.append((i + 1, t))
    return out


def dup_headings(txt):
    """같은 줄이 연달아 두 번 인쇄되는가 (중복 `\\paragraph` 징후)."""
    prev, out = None, []
    for i, l in enumerate(txt.split('\n'), 1):
        t = re.sub(r'^\s*\d+\s*', '', l).strip()
        if (t and len(t) > 12 and t == prev
                and re.search(r'[A-Za-z가-힣]{4,}', t)
                and not re.match(r'^[\d.\s|&+-]+$', t)
                and len(re.findall(r'\d', t)) < len(t) * 0.3):
            out.append((i, t[:70]))
        prev = t
    return out


def dup_paragraph_titles():
    """★`\\paragraph{X}` 가 **연달아 같은 제목**으로 두 번 나오는가 (tex 원본에서).

    ⛔인쇄면만 보면 못 잡는다: `\\paragraph` 는 런인 굵은 글씨라 뒤따르는 본문이 달라
    줄이 같아지지 않는다(항진 검사가 그것을 드러냈다).
    ⟹ 이 하나만 **원본**을 본다. 연속 중복 `\\paragraph` 는 **반드시** 인쇄 손상이다.
    """
    import glob
    out = []
    for rel in ('paper/wavesplat.tex', 'thesis/thesis_main.tex'):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        titles = [(i, m.group(1)) for i, l in
                  enumerate(open(path, encoding='utf-8'), 1)
                  for m in [re.match(r'\\paragraph\{([^}]+)\}', l.strip())] if m]
        for (i1, t1), (i2, t2) in zip(titles, titles[1:]):
            if t1 == t2:
                out.append((rel.split('/')[0], i2, t2))
    return out


def main():
    # ⚠`--paper-only`: pre-commit 훅용. 두 PDF 를 다 읽으면 11 s 이고, 훅이 느리면
    #   사람이 `--no-verify` 로 우회한다. 학위논문은 전체 실행이 본다.
    docs = [(n, r) for n, r in PDFS if not ('--paper-only' in sys.argv and n != 'paper')]
    bad = []
    for doc, ln, t in dup_paragraph_titles():
        bad.append((doc, ln, '중복 paragraph 제목',
                    '같은 제목이 연달아 두 번 (원본 검사)', t[:70]))
    for name, rel in docs:
        txt = text_of(rel)
        if txt is None:
            print(f'  [FAIL] {name}: PDF 를 읽지 못했다 ({rel}) — 검사가 무효다')
            return 1
        lines = txt.split('\n')
        for pname, pat, why in PATTERNS:
            rx = re.compile(pat, re.M)
            for i, l in enumerate(lines, 1):
                if rx.search(l):
                    bad.append((name, i, pname, why, l.strip()[:74]))
        for i, t in dup_headings(txt):
            bad.append((name, i, '연속 중복 줄', '같은 줄이 연달아 인쇄된다', t))
        # ⛔[2026-09-02] `orphan_fragments` 를 **쓰지 않는다**. 두 판 모두 오탐이 지배했다:
        #   ①문단의 마지막 줄 전부(98건) ②한국어 줄바꿈과 참고문헌 항목(18건).
        #   ★`pdftotext` 출력에서 "문단"을 신뢰 있게 복원할 수 없다는 것이 결론이다.
        #   ⟹ **불확실한 검사는 없느니만 못하다**(오늘 세 번 배웠다). 함수는 남기되
        #     호출하지 않고, 왜 안 쓰는지를 여기 적는다.
    print('── 인쇄면 검사 ────────────────────────────────────────')
    if not bad:
        print(f'  [OK  ] {len(PATTERNS) + 2} 패턴 · {len(docs)} 문서 · 손상 0건')
        return 0
    for doc, ln, pname, why, txt in bad[:30]:
        print(f'  [FAIL] {doc}:{ln}  [{pname}] {why}')
        print(f'         {txt}')
    if len(bad) > 30:
        print(f'  … 외 {len(bad) - 30}건')
    print(f'\n⛔{len(bad)}건 — 조판된 문장이 망가졌다.')
    print('⚠오탐이면 PATTERNS 를 **좁혀라**(넓히지 마라 — 늑대를 외치면 무시당한다).')
    return 1


if __name__ == '__main__':
    sys.exit(main())
