"""M5b RESULT 재현 스크립트 — ⛔런 0회. 동결 jsonl 만 읽는다.

`results_m5b_*.jsonl` 를 **`gates_m5b` 를 임포트하지 않고** 두 번째 구현으로
재집계해 `verdict_m5b.json` 과 대조한다. 불일치가 하나라도 있으면 AssertionError.
(본보기 = `tmp/B2_REAL_2026-08-23/depth_recheck.py` · `tmp/STATUS_2026-08-28/f1_table.py`
— 같은 기능이 두 파일에 있고 한쪽만 계측되는 결함 부류에 대한 처방.)

    python3 tmp/M5b_WINDOW_2026-08-28/recheck_verdict.py
"""
import glob
import json
import os
import statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_m5b_*.jsonl'))):
        with open(p) as f:
            rows += [json.loads(x) for x in f if x.strip()]
    return rows


def psign(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def paired(ok, level, ref, arm, field='nowcast', wins=(3,)):
    a = {(r['record'], r['win']): r[field] for r in ok
         if r['level'] == level and r['cell'] == arm and r['win'] in wins}
    b = {(r['record'], r['win']): r[field] for r in ok
         if r['level'] == ref and r['cell'] == arm and r['win'] in wins}
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    rel = st.median([(a[k] - b[k]) / b[k] * 100 for k in ks])
    return dict(n=len(ks), better=w, rel=rel, p=psign(w, len(ks)))


def eq(name, got, exp, tol=1e-9):
    ok = (abs(got - exp) <= tol) if isinstance(exp, float) else (got == exp)
    print(f'  {"OK " if ok else "⛔MISMATCH"} {name}: recomputed={got} verdict={exp}')
    assert ok, f'{name}: {got} != {exp}'


def main():
    rows = load()
    ok = [r for r in rows if 'error' not in r]
    V = json.load(open(os.path.join(HERE, 'verdict_m5b.json')))

    eq('n_rows', len(ok), V['n_rows'])
    eq('errors', len(rows) - len(ok), 0)
    keys = [(r['level'], r['cell'], r['record'], r['win']) for r in ok]
    eq('dup_cells', len(keys) - len(set(keys)), 0)
    eq('label', V['label'], 'WINDOW-LONGER-IS-BETTER')
    eq('gates_all_pass', all(g[1] for g in V['gates']), True)
    eq('n_gates', len(V['gates']), 27)

    for tag, args, vd in [
            ('PRIMARY  PROP TP2', ('TP2', 'ANCHOR', 'PROP'), V['primary']),
            ('판별자   SHIP TP2', ('TP2', 'ANCHOR', 'SHIP'), V['discriminator'])]:
        r = paired(ok, *args)
        eq(f'{tag} better', r['better'], vd['better'])
        eq(f'{tag} n', r['n'], vd['n'])
        eq(f'{tag} rel%', r['rel'], vd['rel_median_pct'])
        eq(f'{tag} p', r['p'], vd['p_one_sided'])

    s = paired(ok, 'TP2', 'ANCHOR', 'PROP', wins=(1, 3, 5))
    eq('n=36 better', s['better'], V['secondary_n36']['better'])
    for d in V['dose_response']:
        r = paired(ok, d['level'], 'ANCHOR', 'PROP')
        eq(f'dose PROP {d["level"]} better', r['better'], d['better'])
    for d in V['extra']['dose_ship']:
        r = paired(ok, d['level'], 'ANCHOR', 'SHIP')
        eq(f'dose SHIP {d["level"]} better', r['better'], d['better'])

    mono = [d['better'] for d in V['dose_response']]
    eq('dose monotone(7>=4>=2)', all(x >= y for x, y in zip(mono, mono[1:])), True)

    # 라벨 분기 재유도: w<=2 (역방향 >= 10/12) ⟹ WINDOW-LONGER-IS-BETTER
    w = paired(ok, 'TP2', 'ANCHOR', 'PROP')['better']
    eq('label branch (w<=2)', w <= 2, True)

    # NOOP 양성대조: win3 에서 ANCHOR 와 비트동일
    for arm in ('SHIP', 'PROP'):
        a = {r['record']: (r['obs_hash'], r['nowcast_bits']) for r in ok
             if r['level'] == 'NOOP' and r['cell'] == arm and r['win'] == 3}
        b = {r['record']: (r['obs_hash'], r['nowcast_bits']) for r in ok
             if r['level'] == 'ANCHOR' and r['cell'] == arm and r['win'] == 3}
        eq(f'NOOP=={arm}-ANCHOR bits', sum(a[k] != b[k] for k in a), 0)

    # 밀도 공변 (공시 전용) — verdict 표와 대조
    for d in V['density_disclosure']:
        v = [r for r in ok if r['level'] == d['level'] and r['win'] == 3
             and r['cell'] == 'PROP']
        eq(f'density {d["level"]} dens_per_s',
           round(st.median([r['dens_per_s'] for r in v]), 4), d['dens_per_s'])

    # 단일요인 되읽기
    eq('n_obs unique', sorted({r['n_obs'] for r in ok}), [600])
    eq('cap_mode unique', sorted({r['cap_mode_used'] for r in ok}), ['end_anchored'])
    eq('t_end==t0 rows', sum(abs(r['t_end_abs'] - r['t0_abs']) > 1e-9 for r in ok), 0)

    print('\n✅recheck_verdict: 전건 일치 (jsonl 재집계 == verdict_m5b.json)')


if __name__ == '__main__':
    main()
