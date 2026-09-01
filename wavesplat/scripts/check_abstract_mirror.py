#!/usr/bin/env python3
"""★국문/영문 초록의 **수치 원자**가 같은가.

⛔왜(2026-09-02):
학위논문 감사가 *"KO/EN abstract split"* 을 지적했고, 실측하니 **국문 초록만 `T-4`
(시간 성분 배제)를 보고**하고 영문에는 그 문단이 **통째로 없었다**. 원자 대조에서
KO 전용 `34/48`·`1.0` 둘로 드러났다.

★이 문서의 tex 주석은 그 위험을 **이미 알고** 있었다 --- *"영문 초록에 같은 문단을
동시에 반영한다"* 가 여러 곳에 적혀 있다. ⟹ 사람의 기억에 맡기던 것을 기계로 옮긴다.

⚠**한계**: 수치 원자만 본다. 서술만 다르고 수가 같으면 못 잡는다.
   그래도 이 결함 부류(한쪽에만 결과를 싣는 것)는 거의 항상 수를 동반한다.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEX = os.path.join(ROOT, 'thesis', 'thesis_main.tex')

# 두 초록의 경계는 `\chapter*{...}` 로 잡는다(줄번호 고정 금지 --- 편집마다 밀린다).
KO_HEAD = r'\chapter*{초\quad 록}'
EN_HEAD = r'\chapter*{Abstract}'
END = r'\chapter{'

# 공백은 두 언어에서 자연히 다르다. 정수 비.퍼센트.소수만 본다.
ATOM = re.compile(r'\d+/\d+|[-−+]?\d+\.\d+\\?%|\d+\.\d+')


def strip_comments(seg):
    out = []
    for line in seg.split('\n'):
        if line.lstrip().startswith('%'):
            continue
        out.append(re.sub(r'(?<!\\)%.*$', '', line))
    return ' '.join(out)


def main():
    s = open(TEX, encoding='utf-8').read()
    for h in (KO_HEAD, EN_HEAD):
        if h not in s:
            print(f'⛔초록 머리를 찾지 못했다: {h!r} — 이 검사는 무효다')
            return 1
    ko0, en0 = s.index(KO_HEAD), s.index(EN_HEAD)
    ko1 = en0
    en1 = s.index(END, en0)
    ko, en = ATOM.findall(strip_comments(s[ko0:ko1])), ATOM.findall(strip_comments(s[en0:en1]))
    K, E = set(ko), set(en)
    only_k, only_e = sorted(K - E), sorted(E - K)
    print('── 초록 KO/EN 원자 대조 ────────────────────────────────')
    print(f'  KO {len(K)}종 · EN {len(E)}종 · 공통 {len(K & E)}종')
    if not only_k and not only_e:
        print('  [OK  ] 한쪽에만 있는 수치 원자 0건')
        return 0
    if only_k:
        print(f'  [FAIL] 국문에만: {only_k}')
    if only_e:
        print(f'  [FAIL] 영문에만: {only_e}')
    print('\n⛔한쪽 초록에만 있는 결과가 있다 --- 두 초록은 같은 것을 보고해야 한다.')
    print('⚠의도한 비대칭이라면 이 스크립트에 사유와 함께 예외를 적어라.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
