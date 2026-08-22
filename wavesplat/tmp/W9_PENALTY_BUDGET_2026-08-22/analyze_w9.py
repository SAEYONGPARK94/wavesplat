"""W7 분석기 — 결과 파일보다 **먼저** 쓰였다 (S4·S5 가 어긴 조건).

★게이트·팔·판정 규칙을 하나도 재기재하지 않는다. 전부 `gates_w9` 에서 온다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_w9 as G  # noqa: E402

OUT = os.environ.get('W9_OUT', os.path.join(HERE, 'results.jsonl'))


def main():
    rows = [json.loads(l) for l in open(OUT)]
    ok = [r for r in rows if 'error' not in r]

    print('=' * 78)
    print('W9 — 척도불변 벌점 개입 (규칙은 gates_w9 단일출처)')
    print('=' * 78)

    print('\n## 게이트 (판정 전)')
    gs = G.gates(rows)
    abort = False
    for gid, good, msg in gs:
        print(f'  [{"PASS" if good else "FAIL"}] {gid}: {msg}')
        if not good and 'ABORT' in gid:
            abort = True
    if abort:
        print('\n⛔ ABORT 게이트 실패 — 사전등록 §4 에 따라 판정하지 않는다.')
        return
    if not all(g[1] for g in gs):
        print('\n⚠ 게이트 실패 있음 — 판정은 보류 등급으로 읽어라.')

    print(f'\n## 기술통계 ({G.REGIME}, n={G.N_EPISODES})')
    import statistics as st
    for arm in G.ARM_ORDER:
        v = [r[G.FIELD] for r in ok if r['cell'] == arm]
        si = {r.get('scale_invariant_readback') for r in ok if r['cell'] == arm}
        print(f'  {arm:18s} SI={si}  창안 중앙 {st.median(v):.6f}')

    print('\n## 대조 (짝, vs ORA-WPHI-0STEP)')
    for arm in ('ORA-WPHI-SHIP', 'ORA-WPHI-SI'):
        v = G.ratio_vs_ref(ok, arm)
        print(f'    {arm:18s} 진실보다 나쁨 {v["above"]}/{v["n"]}  '
              f'ratio {v["ratio"]:.4f}  p(단측) {v["p_one_sided"]:.5f}')

    primary = G.ratio_vs_ref(ok, 'ORA-WPHI-SI')
    lab, meaning = G.label(primary)
    print('\n' + '=' * 78)
    print(f'## PRIMARY = {primary["wins"]}/{primary["n"]} '
          f'(문턱 {G.THRESHOLD})  ratio {primary["ratio"]:.4f}  '
          f'p={primary["p_one_sided"]:.5f}')
    print(f'## 판정 = {lab}')
    print(f'##   {meaning}')
    print('=' * 78)

    json.dump({'primary': primary, 'label': lab, 'meaning': meaning,
               'gates': [{'id': g, 'ok': o, 'msg': m} for g, o, m in gs],
               'arms': {a: G.ratio_vs_ref(ok, a)
                        for a in ('ORA-WPHI-SHIP', 'ORA-WPHI-SI')}},
              open(os.path.join(HERE, 'verdict_w7.json'), 'w'),
              ensure_ascii=False, indent=2)
    print('\nverdict_w7.json 기록')


if __name__ == '__main__':
    main()
