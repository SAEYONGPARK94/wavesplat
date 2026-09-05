#!/usr/bin/env python3
"""`AMENDMENT_4` 증명 — **「공시 말고는 한 비트도 안 움직였다」**.

⛔판정 도구가 **아니다**. 새 런 0회 · 추정기 0회. 하는 일은 하나:
개정 **전** `verdict_t13.json`(git 판본)과 **현재** 판본을 **키별 바이트 대조**하고,
움직인 키가 선언된 둘(`disclosures`·`amendments`) 뿐임을 단언한다.

    python3 tmp/T13_DEPTHFIX_2026-09-04/verify_amendment4.py [<git-rev>]

기본 rev = `59e4b04`(개정문 3 커밋 = 개정 전 마지막 판본).
⚠`check_batch_artifacts._judge_files()` 의 사후 포렌식 예외에 해당한다
(`gates*`·`run*`·`analyze*` 가 아니며 판정을 낳지 않는다 — `B2_REAL/depth_recheck.py` 선례).

★「사라졌다」와 「확장됐다」를 **접두사 길이 같은 임의 문턱으로 가르지 않는다** —
초판이 정확히 그 방식으로 `wall_s` 공시를 「있는데 없다」고 오판했다(이 저장소가
반복해 대가를 치른 부류: *검사기가 「있는 것을 없다고」 하는 것도 그 반대만큼 나쁘다*).
대신 옛 문자열이 새 문자열의 **부분수열(삽입만)** 인지를 본다 = 삭제 0자의 기계 증명.
"""
import difflib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.relpath(os.path.join(HERE, 'verdict_t13.json'),
                      os.path.abspath(os.path.join(HERE, '..', '..', '..')))
DEFAULT_REV = '59e4b04'

# ⛔의도적으로 바뀌는 두 키. 그 밖이 하나라도 움직이면 실패한다.
DECLARED = {'disclosures', 'amendments'}


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REV
    root = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
    old = json.loads(subprocess.run(['git', 'show', f'{rev}:{REL}'], cwd=root,
                                    capture_output=True, text=True, check=True).stdout)
    new = json.load(open(os.path.join(HERE, 'verdict_t13.json'), encoding='utf-8'))

    assert set(old) == set(new), f'키 집합이 다르다: {set(old) ^ set(new)}'
    same, moved = [], []
    for k in old:
        a = json.dumps(old[k], ensure_ascii=False, sort_keys=True)
        b = json.dumps(new[k], ensure_ascii=False, sort_keys=True)
        (same if a == b else moved).append(k)

    print(f'rev {rev} 대비 · 키 {len(old)}개 · **비트 동일 {len(same)}** · 움직임 '
          f'{len(moved)} {sorted(moved)}')
    print('── 비트 동일 (판정·게이트·수치·p값·중앙값) ──')
    print('  ' + ' · '.join(sorted(same)))
    bad = sorted(k for k in moved if k not in DECLARED)
    if bad:
        print(f'⛔선언되지 않은 키가 움직였다: {bad}')
        return 1

    go, gn = old['gates'], new['gates']
    assert len(go) == len(gn), '게이트 수가 다르다'
    nd = sum((x['name'], x['pass_'], x['detail']) != (y['name'], y['pass_'], y['detail'])
             for x, y in zip(go, gn))
    print(f'── 게이트 {len(go)}개 이름·pass_·detail 전건 대조 불일치 {nd} · '
          f'gates_pass {new["gates_pass"]} · label {new["label"]!r} ──')

    print('── amendments (선언된 변경) ──')
    for x in new['amendments']:
        print(('  [+] ' if x not in old['amendments'] else '  [ ] ') + x)
    assert all(x in new['amendments'] for x in old['amendments']), '기존 개정문 항목 소실'

    o, n = old['disclosures'], new['disclosures']
    print(f'── disclosures {len(o)} → {len(n)} ──')
    changed = [x for x in o if x not in n]
    for x in changed:
        hit = None
        for y in n:
            ops = difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes()
            if all(t in ('equal', 'insert') for t, *_ in ops):
                hit, ins = y, ''.join(y[c:d] for t, a, b, c, d in ops if t == 'insert')
                break
        assert hit is not None, f'⛔공시가 사라졌다(삽입만으로 설명 안 됨): {x[:80]}'
        print(f'  [확장] {len(x)}자 → {len(hit)}자 · **삭제 0자 · 삽입 {len(ins)}자** '
              f'· 옛: {x[:56]}…')
    for y in n:
        if y not in o and not any(
                all(t in ('equal', 'insert') for t, *_ in
                    difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes())
                for x in changed):
            print(f'  [신설] {y[:78]}…')
    print()
    print('✅증명: 판정·게이트·수치 전부 비트 동일 · 공시는 **삭제 0자 순수 가산**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
