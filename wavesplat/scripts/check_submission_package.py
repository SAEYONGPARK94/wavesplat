#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★**제출 패키지 정합성 검사기** — 원고가 아니라 **보내는 것 전체**를 본다.

⛔왜 (2026-09-02, `N-1` 토론):
> *"「제출 준비됨」의 단위는 원고가 아니라 **패키지**다."*

`check_paper_disclosures.py` 는 `paper/waveprior.tex` **한 파일만** 보고, 인용금지 래칫은
금지 **수치**만 본다. 그래서 **커버레터를 보는 기계가 0개**였고, 실측하면 커버레터가
원고와 여섯 곳에서 어긋나 있었다 --- 그중 둘은 **편집장에게 하는 사실 주장**이다:

  · 제목이 여전히 옛 것 · **철회된 기여**를 판다 · `[RETRACTED]` 스텁을 편지에 싣는다
  · *"데이터는 github 에 공개돼 있다"* ⟺ 원고 §Data availability 는 *"채택 시 기탁"*
  · 추천 심사자를 **그날 거짓으로 판명된 귀속**을 근거로 추천

★그리고 배너(`DO NOT SUBMIT`)는 **사람이 읽는 주석**이라 이것을 막지 못했다.
이 검사기가 배너가 하던 일을 **기계로** 한다.

사용: `python3 scripts/check_submission_package.py [--selftest]`
종료코드 1 = 패키지가 원고와 어긋난다.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEX = os.path.join(ROOT, 'paper', 'waveprior.tex')
LETTERS = ['paper/cover_letter.tex', 'paper/submission/cover_letter.tex']


def body(path):
    """주석 제거 + 공백 정규화. ⛔주석은 자구를 **인용해서** 설명하므로 세지 않는다."""
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path, encoding='utf-8'):
        if line.lstrip().startswith('%'):
            continue
        out.append(re.sub(r'(?<!\\)%.*$', '', line))
    return re.sub(r'\s+', ' ', ' '.join(out))


def manuscript_title(t):
    m = re.search(r'(?m)^\\title\{(.+?)\}\s*$', t, re.S)
    return re.sub(r'\s+', ' ', m.group(1).replace('\\\\', ' ')).strip() if m else None


def norm(s):
    """비교용 정규화 — LaTeX 장식·구두점·대소문자를 지운다."""
    s = re.sub(r'\\[A-Za-z]+\*?', ' ', s)
    s = re.sub(r'[{}`\'"“”‘’~,.:;\-–—]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def latest_frozen_batches():
    """**가장 최근 날짜**의 배치 디렉터리에서 라벨을 가진 verdict 를 모은다.

    ⚠[2026-09-02] 초판은 `mtime` 으로 「최신」을 정했다가 오래된 배치를 세 개 올렸다
    (파일을 건드리면 mtime 이 움직인다) ⟹ **디렉터리 이름의 날짜**를 쓴다.
    """
    by_date = {}
    for p in glob.glob(os.path.join(ROOT, 'tmp', '*', 'verdict_*.json')):
        d = os.path.basename(os.path.dirname(p))
        m = re.search(r'_(\d{4}-\d{2}-\d{2})$', d)
        if not m:
            continue
        try:
            v = json.load(open(p))
        except Exception:
            continue
        lab = v.get('label')
        if not lab:
            continue
        tag = v.get('batch') or d.split('_')[0]
        by_date.setdefault(m.group(1), {})[str(tag)] = lab
    if not by_date:
        return {}
    newest = max(by_date)
    return {t: (newest, lab) for t, lab in by_date[newest].items()}


def checks():
    bad, fired = [], []
    tex = body(TEX)
    if tex is None:
        return [('원고 부재', f'{TEX} 가 없다 — 검사가 무효다')], fired
    title = manuscript_title(open(TEX, encoding='utf-8').read())

    for rel in LETTERS:
        b = body(os.path.join(ROOT, rel))
        if b is None:
            continue
        fired.append(rel)
        # ① 제목 일치
        if title and norm(title)[:60] not in norm(b):
            bad.append((f'{rel} 제목 불일치',
                        f'원고 제목이 커버레터에 없다: "{title[:64]}…"'))
        # ② 철회 스텁이 편지에 실린다
        n = len(re.findall(r'RETRACTED', b))
        if n:
            bad.append((f'{rel} 철회 스텁 {n}건',
                        '⛔편집장에게 보내는 편지에 [RETRACTED] 를 싣지 않는다'))
        # ③ 데이터 가용성 모순 — 원고가 "채택 시" 인데 편지가 "이미 공개" 라 말한다
        upon = re.search(r'upon acceptance', tex, re.I)
        openly = re.search(r'(openly|publicly) available at', b, re.I)
        if upon and openly:
            bad.append((f'{rel} 데이터 가용성 모순',
                        '원고 = "upon acceptance" · 편지 = "openly available at" '
                        '⟹ 편집장에게 하는 사실 주장이 어긋난다'))

    # ④ 논문이 최신 동결 배치를 인용하는가
    frozen = latest_frozen_batches()
    if frozen:
        fired.append('최신 동결 배치')
        for tag, (_, lab) in sorted(frozen.items()):
            pat = re.escape(tag).replace(r'\-', '[-‑]?')
            if not re.search(pat, tex, re.I):
                bad.append((f'최신 배치 `{tag}` 미반영',
                            f'동결 라벨 "{str(lab)[:44]}…" 이 원고에 0회'))
    return bad, fired


def selftest():
    """★위조 검사 — 규칙이 아무것도 막지 않으면 그것을 드러낸다."""
    bad, fired = checks()
    print('── 제출 패키지 항진 검사 ' + '─' * 26)
    print(f'  발동 대상 {len(fired)}: {", ".join(fired)}')
    print(f'  현재 위반 {len(bad)}건 — ★0 건이면 규칙이 **미검증**이다(통과로 세지 마라)')
    for n, w in bad:
        print(f'    · {n}')
    ok = len(fired) >= 2
    print('  ' + ('[OK  ] 규칙이 실제 대상에서 발동한다' if ok
                  else '[FAIL] 검사 대상을 하나도 찾지 못했다'))
    return 0 if ok else 1


def main():
    if '--selftest' in sys.argv:
        return selftest()
    bad, fired = checks()
    print('── 제출 패키지 정합성 ' + '─' * 30)
    for n, w in bad:
        print(f'  ⛔{n}: {w}')
    if bad:
        print(f'  [FAIL] {len(bad)}건 — ★원고가 아니라 **보내는 것**이 어긋나 있다.')
        return 1
    print(f'  [OK  ] 대상 {len(fired)} · 어긋남 0건')
    return 0


if __name__ == '__main__':
    sys.exit(main())
