#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★재실행이 **계수를 바꾸지 않았는가** --- 보존본과 키별 비트 대조. ⛔런 0회.

`WHY_RERUN.md` 가 결과 전에 약속한 검사다:

> 재실행 행을 보존본과 **키별로 대조**한다. 추정이 `(episode, arm, seed)` 에 결정적이면
> 겹치는 행이 **비트 동일**해야 한다. 동일하면 *"2-PID 결함은 계수를 한 자리도 바꾸지
> 않았다"* 가 **자료로** 참이 된다. ⛔다르면 그 사실이 훨씬 중요한 발견이므로
> **그것을 먼저 보고**한다.

⚠보존본(`ABORTED_2PID_rows.jsonl`)은 **판정에 쓰이지 않는다**. 이 스크립트는
그것을 오직 **무결성 대조군**으로만 읽는다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ('episode', 'arm', 'seed')
# ⚠런마다 정당하게 달라지는 열은 대조에서 뺀다(그 사실 자체를 인쇄한다).
VOLATILE = ('wall_s', 'pid', 'git_head', 'init_time_s')


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            out[tuple(r.get(k) for k in KEY)] = r
    return out


def main():
    new = load('results_t9.jsonl')
    old = load('ABORTED_2PID_rows.jsonl')
    if new is None or old is None:
        print('⛔대조할 파일이 없다')
        return 2
    both = sorted(set(new) & set(old))
    print(f'재실행 {len(new)}행 · 보존본 {len(old)}행 · 겹침 {len(both)}행')
    print(f'⚠대조에서 뺀 열(런마다 정당하게 다르다): {", ".join(VOLATILE)}')
    diff = []
    for k in both:
        a = {x: v for x, v in new[k].items() if x not in VOLATILE}
        b = {x: v for x, v in old[k].items() if x not in VOLATILE}
        if a != b:
            keys = sorted({x for x in set(a) | set(b) if a.get(x) != b.get(x)})
            diff.append((k, keys))
    if not diff:
        print(f'✅겹치는 {len(both)}행이 **전건 동일** '
              f'⟹ 2-PID 결함은 계수를 한 자리도 바꾸지 않았다.')
        return 0
    print(f'⛔{len(diff)}행이 다르다 --- ★이것이 훨씬 중요한 발견이다. 먼저 보고하라.')
    for k, keys in diff[:10]:
        print(f'   {k}  다른 열: {keys[:6]}')
        for x in keys[:3]:
            print(f'       {x}: 재실행 {new[k].get(x)!r} vs 보존 {old[k].get(x)!r}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
