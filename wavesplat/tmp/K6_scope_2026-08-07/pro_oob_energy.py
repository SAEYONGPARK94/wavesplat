"""PRO (6): are the out-of-band atoms INERT (harmless) or ENERGETIC (harmful)?
Uses the dumped per-atom (kx, ky, w). GT-free -- no truth is read."""
import json
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
CELLS = ['X-BASE', 'G1', 'G1P1', 'AUTO', 'ORACLE-K']
KCUT = 0.8

print('=' * 112)
print('Out-of-band atoms (|k| > 0.8 rad/m, i.e. T < 2.24 s at d=50 m): COUNT and ENERGY SHARE')
print('  Eoob% = sum(w^2) over |k|>0.8  /  sum(w^2) over all atoms   [median of 3 seeds]')
print('  wmax_oob / wmax_in = largest single amplitude out-of-band vs in-band')
print('=' * 112)
print(f'{"episode":30s}{"cell":9s}{"n_oob":>7s}{"n_tot":>7s}{"Eoob%":>8s}'
      f'{"wmax_oob":>10s}{"wmax_in":>9s}')
for eps, nm in [(CROWDED4, 'CROWDED'), (NEW8, 'SPARSE')]:
    print(f'---- {nm} ----')
    for e in eps:
        for c in CELLS:
            noob, ntot, eo, wo, wi = [], [], [], [], []
            for r in by[(c, e)]:
                k = np.hypot(np.asarray(r['kx']), np.asarray(r['ky']))
                w = np.abs(np.asarray(r['w']))
                if not k.size: continue
                m = k > KCUT
                noob.append(int(m.sum())); ntot.append(int(k.size))
                eo.append(float((w[m] ** 2).sum() / max((w ** 2).sum(), 1e-30)))
                wo.append(float(w[m].max()) if m.any() else 0.0)
                wi.append(float(w[~m].max()) if (~m).any() else 0.0)
            if not ntot: continue
            print(f'{e[8:-3] if c==CELLS[0] else "":30s}{c:9s}{np.median(noob):7.0f}'
                  f'{np.median(ntot):7.0f}{np.median(eo)*100:8.2f}'
                  f'{np.median(wo):10.4f}{np.median(wi):9.4f}')
        print()

print('=' * 112)
print('BLOCK SUMMARY — median over episodes')
print('=' * 112)
print(f'{"block":10s}{"cell":10s}{"n_oob":>8s}{"Eoob%":>9s}')
for eps, nm in [(CROWDED4, 'CROWDED'), (NEW8, 'SPARSE')]:
    for c in CELLS:
        A, B = [], []
        for e in eps:
            no, ee = [], []
            for r in by[(c, e)]:
                k = np.hypot(np.asarray(r['kx']), np.asarray(r['ky']))
                w = np.abs(np.asarray(r['w']))
                if not k.size: continue
                m = k > KCUT
                no.append(int(m.sum()))
                ee.append(float((w[m] ** 2).sum() / max((w ** 2).sum(), 1e-30)))
            A.append(np.median(no)); B.append(np.median(ee))
        print(f'{nm:10s}{c:10s}{np.median(A):8.1f}{np.median(B)*100:9.3f}')
