"""PRO (6): (1) m = decorr + shrink decomposition per cell; (2) GT-FREE statistics that a
deployed system could compute -- how many FINAL atoms sit outside a physically admissible
|k| box implied by the dispersion relation + the operating point (no ground truth used)."""
import json, statistics as st
from collections import defaultdict
import numpy as np

B2 = '/home/vexit/wavesplat/wavesplat/tmp/B2_g1p1_ext_2026-08-06/results.jsonl'
rows = [json.loads(l) for l in open(B2)]
by = defaultdict(list)
for r in rows: by[(r['cell'], r['episode'])].append(r)
CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
CELLS = ['X-BASE', 'G1', 'P1', 'G1P1', 'AUTO', 'ORACLE-K']
def M(c, e, f):
    v = [f(r) for r in by[(c, e)] if f(r) == f(r)]
    return float(np.median(v)) if v else float('nan')

print('=' * 106)
print('1. ERROR DECOMPOSITION  m = (1-rho^2) + (sqrt(VR)-rho)^2   [decorr = k/phase misfit,'
      ' shrink = amplitude]')
print('=' * 106)
for blk, eps in [('CROWDED4', CROWDED4), ('SPARSE8', NEW8)]:
    print(f'\n{blk}')
    print(f'  {"cell":10s}{"m":>10s}{"decorr":>10s}{"shrink":>10s}{"decorr%":>10s}{"nRMSE":>10s}')
    for c in CELLS:
        m = st.median([M(c, e, lambda r: r['m_mean']) for e in eps])
        d = st.median([M(c, e, lambda r: r['decorr_mean']) for e in eps])
        s = st.median([M(c, e, lambda r: r['shrink_mean']) for e in eps])
        n = st.median([M(c, e, lambda r: r['nRMSE']) for e in eps])
        print(f'  {c:10s}{m:10.4f}{d:10.4f}{s:10.4f}{d/max(m,1e-9)*100:9.1f}%{n:10.4f}')

# ---------- GT-free atom-location statistic ----------
G = 9.81
def k_of_T(T, depth=50.0):
    w = 2 * np.pi / T
    k = w * w / G
    for _ in range(60):
        k = w * w / (G * np.tanh(k * depth))
    return k

print('\n' + '=' * 106)
print('2. GT-FREE ADMISSIBLE-|k| BOX from the dispersion relation (depth 50 m)')
print('=' * 106)
for T in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0):
    print(f'   wave period {T:4.1f} s  ->  |k| = {k_of_T(T):.4f} rad/m   '
          f'(lambda = {2*np.pi/k_of_T(T):7.2f} m)')
print(f'\n   shipped coarse box  +-4.0 rad/m  admits down to T = 1.0 s (lambda 1.6 m)')
print(f'   G1 NARROW box       +-0.8 rad/m  admits down to T ~ 2.24 s')
print('   NOTE: this bound uses ONLY the dispersion relation + a minimum-period prior.'
      '  NO ground truth.')

print('\n' + '=' * 106)
print('3. WHERE DO THE FINAL ATOMS ACTUALLY SIT?  (from the dumped kx/ky -- GT-free statistic)')
print('   frac|k|>0.8 = fraction of final atoms at |k| > 0.8 (T < 2.24 s == not a wind sea here)')
print('=' * 106)
print(f'{"episode":30s}' + ''.join(f'{c:>18s}' for c in CELLS))
for eps, nm in [(CROWDED4, 'CROWDED'), (NEW8, 'SPARSE')]:
    print(f'-- {nm} --')
    for e in eps:
        line = f'{e[8:-3]:30s}'
        for c in CELLS:
            fr, mx = [], []
            for r in by[(c, e)]:
                kk = np.hypot(np.asarray(r['kx']), np.asarray(r['ky']))
                if kk.size:
                    fr.append(float((kk > 0.8).mean())); mx.append(float(kk.max()))
            line += f'{np.median(fr)*100:11.0f}%/{np.median(mx):5.2f}'
        print(line)
print('\n  (each cell = "% of final atoms with |k|>0.8" / "max |k| of any final atom")')
