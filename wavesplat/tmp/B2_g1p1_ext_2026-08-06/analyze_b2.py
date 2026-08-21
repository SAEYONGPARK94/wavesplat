"""B2 scorer — implements PREREG.md §2 conventions verbatim. Run from wavesplat/ root.

Conventions (frozen before results):
  seed -> episode : MEDIAN of 3
  level           : median of episode medians
  effect          : PAIRED per-episode delta (cell - X-BASE), MEDIAN
  wins            : paired sign test, TWO-SIDED
  recovery        : per-episode ratio (XB-cell)/(XB-ORACLE-K), MEDIAN
  interaction     : convention A (median of per-episode interactions), B/C reported too
"""
import json, math, os, statistics as st
from collections import defaultdict

HERE = 'tmp/B2_g1p1_ext_2026-08-06/'
RX2 = 'tmp/rx2_init_2026-08-05/results.jsonl'
BASELINES = 'tmp/rerun_2026-07-25/batchA_baselines.jsonl'

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
ALL12 = NEW8 + CROWDED4
CELLS = ['X-BASE', 'G1', 'P1', 'G1P1', 'ORACLE-K', 'AUTO']


def sign_p(k, n):
    """two-sided sign test on n non-tied pairs with k wins"""
    if n == 0:
        return 1.0
    kk = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(kk + 1)) / 2 ** n)


def load(path):
    d = defaultdict(dict)
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'nRMSE' in r and 'error' not in r:
            d[(r['cell'], r['episode'])][r['seed']] = r['nRMSE']
    return d


def main():
    cur = load(HERE + 'results.jsonl')
    med = lambda c, e: st.median(list(cur[(c, e)].values())) if cur[(c, e)] else None
    nulls = {r['episode']: r['zero_null_tracking_nRMSE_mean']
             for r in json.load(open(HERE + 'nulls12.json'))}
    letkf = {}
    for line in open(BASELINES):
        if not line.strip():
            continue
        r = json.loads(line)
        if r['method'] == 'LETKF':
            letkf[r['episode']] = float(r['nRMSE'])

    print('=' * 96)
    print('B2 — G1P1 extension 4ep -> 12ep    (PREREG.md conventions, frozen before results)')
    print('=' * 96)

    # ---- coverage ----
    miss = [(c, e) for c in CELLS for e in ALL12 if len(cur[(c, e)]) != 3]
    print(f'\ncoverage: {sum(len(cur[(c,e)]) for c in CELLS for e in ALL12)}/{len(CELLS)*12*3} runs'
          f' | incomplete cells: {len(miss)}')
    if miss:
        print('  INCOMPLETE:', miss[:8])
        return

    # ---- SECONDARY-2: crowded-4 reproduction control (deterministic, no p) ----
    old = load(RX2)
    print('\n-- SECONDARY-2  crowded-4 reproduction vs rx2 (DETERMINISTIC, no p-value) --')
    worst = 0.0
    ncmp = 0
    for c in CELLS:
        for e in CROWDED4:
            for s, v in cur[(c, e)].items():
                if s in old[(c, e)]:
                    worst = max(worst, abs(v - old[(c, e)][s]))
                    ncmp += 1
    print(f'   compared {ncmp} runs | max |diff| = {worst:.3e}'
          f'  ->  {"BIT-IDENTICAL" if worst == 0 else "MISMATCH — HOLD RESULTS"}')

    # ---- per-episode table ----
    print('\n-- per-episode medians (median-of-3 seeds) --')
    hdr = f'{"episode":34s}' + ''.join(f'{c:>10s}' for c in CELLS) + f'{"null":>9s}{"LETKF":>9s}'
    print(hdr)
    for e in ALL12:
        tag = e[8:-3]
        row = f'{tag:34s}' + ''.join(f'{med(c,e):10.4f}' for c in CELLS)
        row += f'{nulls[e]:9.4f}{letkf.get(e,float("nan")):9.4f}'
        print(row)

    def block(name, eps):
        n = len(eps)
        print(f'\n{"="*96}\n{name}   (n={n}, two-sided sign-test floor p = {2/2**n:.3e})\n{"="*96}')
        print(f'{"cell":10s}{"medOfMed":>10s}{"pairedD":>10s}{"wins":>8s}{"p":>10s}'
              f'{"recovery":>10s}{"vsLETKF":>9s}{"vsNULL":>8s}')
        for c in CELLS:
            lv = st.median([med(c, e) for e in eps])
            if c == 'X-BASE':
                d = w = p = rec = None
            else:
                ds = [med(c, e) - med('X-BASE', e) for e in eps]
                d = st.median(ds)
                nz = [x for x in ds if x != 0]
                w = sum(1 for x in nz if x < 0)
                p = sign_p(w, len(nz))
                rec = st.median([(med('X-BASE', e) - med(c, e)) /
                                 (med('X-BASE', e) - med('ORACLE-K', e)) for e in eps])
            vl = sum(1 for e in eps if e in letkf and med(c, e) < letkf[e])
            vn = sum(1 for e in eps if med(c, e) < nulls[e])
            print(f'{c:10s}{lv:10.4f}'
                  + (f'{d:+10.4f}{w:>4d}/{len(nz):<3d}{p:10.4f}{rec*100:9.1f}%'
                     if d is not None else f'{"—":>10s}{"—":>8s}{"—":>10s}{"—":>10s}')
                  + f'{vl:>5d}/{len(eps):<3d}{vn:>4d}/{len(eps):<3d}')

    block('PRIMARY — NEW 8 episodes (sparse/moderate; G1P1 never developed here)', NEW8)
    block('SECONDARY-1 — ALL 12 episodes', ALL12)
    block('context — crowded 4 (already measured by rx2; not new information)', CROWDED4)

    # ---- SECONDARY-3: interaction, convention A primary ----
    print(f'\n{"="*96}\nSECONDARY-3 — interaction  (G1P1-XB) - (G1-XB) - (P1-XB)\n{"="*96}')
    for name, eps in [('NEW8', NEW8), ('ALL12', ALL12), ('crowded4', CROWDED4)]:
        ia = [(med('G1P1', e) - med('X-BASE', e)) - (med('G1', e) - med('X-BASE', e))
              - (med('P1', e) - med('X-BASE', e)) for e in eps]
        A = st.median(ia)
        B = (st.median([med('G1P1', e) - med('X-BASE', e) for e in eps])
             - st.median([med('G1', e) - med('X-BASE', e) for e in eps])
             - st.median([med('P1', e) - med('X-BASE', e) for e in eps]))
        C = (st.mean([med('G1P1', e) - med('X-BASE', e) for e in eps])
             - st.mean([med('G1', e) - med('X-BASE', e) for e in eps])
             - st.mean([med('P1', e) - med('X-BASE', e) for e in eps]))
        print(f'  {name:9s} A(paired median) {A:+.4f}  [neg {sum(1 for x in ia if x<0)}/{len(ia)}]'
              f'   B(diff of medians) {B:+.4f}   C(mean) {C:+.4f}')

    # ---- strata ----
    print(f'\n{"="*96}\nSTRATA — paired delta (cell - X-BASE), median  [descriptive only]\n{"="*96}')
    def nc(e):
        import re
        return int(re.search(r'_(\d+)comp', e).group(1))
    strata = {
        'ncomp 1-2   (n=4)': [e for e in ALL12 if nc(e) <= 2],
        'ncomp 5     (n=4)': [e for e in ALL12 if nc(e) == 5],
        'ncomp 15-30 (n=4)': [e for e in ALL12 if nc(e) >= 15],
        'uniform     (n=10)': [e for e in ALL12 if e.endswith('uniform.h5')],
        'slope       (n=2)': [e for e in ALL12 if e.endswith('slope.h5')],
    }
    print(f'{"stratum":22s}' + ''.join(f'{c:>12s}' for c in CELLS if c != 'X-BASE'))
    for name, eps in strata.items():
        cells = [c for c in CELLS if c != 'X-BASE']
        vals = []
        for c in cells:
            ds = [med(c, e) - med('X-BASE', e) for e in eps]
            vals.append(f'{st.median(ds):+8.4f} {sum(1 for x in ds if x<0)}/{len(ds)}')
        print(f'{name:22s}' + ''.join(f'{v:>12s}' for v in vals))

    # ---- verdict against the frozen bar ----
    ds = [med('G1P1', e) - med('X-BASE', e) for e in NEW8]
    dmed = st.median(ds)
    w = sum(1 for x in ds if x < 0)
    print(f'\n{"="*96}\nFROZEN BAR (PREREG §3 PRIMARY, NEW8)\n{"="*96}')
    print(f'  paired delta median = {dmed:+.4f}   wins = {w}/8   two-sided p = {sign_p(w,8):.4f}')
    if dmed <= 0 and w >= 6:
        v = 'PASS'
    elif dmed > 0 or w <= 4:
        v = 'FAIL'
    else:
        v = 'INCONCLUSIVE'
    print(f'  PASS iff (delta<=0 AND wins>=6/8) | FAIL iff (delta>0 OR wins<=4/8) | else INCONCLUSIVE')
    print(f'  ==> VERDICT: {v}')


if __name__ == '__main__':
    main()
