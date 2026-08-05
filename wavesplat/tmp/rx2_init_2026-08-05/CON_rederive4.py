"""CON part 4 — the decomposition that replaces "3.88x".

Cross-experiment comparison is licensed by the bit-identity check in CON_rederive3.txt:
E2 cell A0 reproduces s1s2 arm S1-A0 to +0.0000 on all four crowded episodes (same
seeds 42/123/456, same episodes, same config).
"""
import json, os
import numpy as np
from math import comb

TMP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SNAPD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasnap')
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


def load(p, key):
    rows = []
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r:
            r['_cell'] = r[key]; rows.append(r)
    return rows


def sign_p(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else float('nan')


def lv(rows, cell, ep, m):
    v = [r[m] for r in rows if r['_cell'] == cell and r['episode'] == ep
         and np.isfinite(r.get(m, np.nan))]
    return float(np.median(v)) if v else float('nan')


CROWD = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
         'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
s1s2 = load(os.path.join(SNAPD, 's1s2_2026-08-05.results.jsonl'), 'arm')
e2 = load(os.path.join(SNAPD, 'e2_grid_sep_2026-08-05.results.jsonl'), 'cell')
e1 = load(os.path.join(SNAPD, 'e1_grid_2026-08-05.results.jsonl'), 'cell')

P('=' * 104)
P('THE LADDER — init_nrmse on the 4 crowded episodes (median-of-3), every arm ever run')
P('=' * 104)
ARMS = [
    ('shipped ladder [3,8,15]',      s1s2, 'S1-A0',  'deployable'),
    ('budget 40 (S1-A3)',            s1s2, 'S1-A3',  'deployable'),
    ('N=N_gt, beamform (S2-C1)',     s1s2, 'S2-C1',  'deployable'),
    ('E1-B  narrow k-range',         e1,   'B',      'deployable'),
    ('E1-D  narrow + recompute',     e1,   'D',      'deployable'),
    ('E2-U0 grid  81, radius 0.200', e2,   'U0',     'deployable'),
    ('E2-U1 grid 401, radius 0.200', e2,   'U1',     'deployable'),
    ('E2-U3 grid  81, radius 0.100', e2,   'U3',     'deployable'),
    ('E2-U2 grid 401, radius 0.100', e2,   'U2',     'deployable'),
    ('E2-U1M2 U1 + min-sep 0.02',    e2,   'U1M2',   'deployable'),
    ('ORACLE k, k trainable (T1)',   s1s2, 'S2-T1',  'ORACLE'),
    ('ORACLE k, k FROZEN (T1f)',     s1s2, 'S2-T1f', 'ORACLE+freeze'),
]
P(f'{"arm":32s} {"class":14s} ' + ' '.join(f'{e.split("_")[1]+"_"+e.split("_")[2]:>13s}'
                                           for e in CROWD) + f' {"MEDIAN":>8s} {"MEAN":>8s}')
tab = {}
for name, rows, c, cls in ARMS:
    vals = [lv(rows, c, e, 'init_nrmse') for e in CROWD]
    tab[name] = vals
    P(f'{name:32s} {cls:14s} ' + ' '.join(f'{v:13.4f}' for v in vals) +
      f' {np.median(vals):8.4f} {np.mean(vals):8.4f}')

P('')
ship = np.array(tab['shipped ladder [3,8,15]'])
best_dep = min([a for a in ARMS if a[3] == 'deployable'], key=lambda a: np.median(tab[a[0]]))
bd = np.array(tab[best_dep[0]])
t1 = np.array(tab['ORACLE k, k trainable (T1)'])
t1f = np.array(tab['ORACLE k, k FROZEN (T1f)'])
a3 = np.array(tab['budget 40 (S1-A3)'])

P('=' * 104)
P('DECOMPOSITION of the quoted "3.88x / 0.1809" init gap')
P('=' * 104)
P(f'  reference used by the claim : best deployable = S1-A3, median init {np.median(a3):.4f}')
P(f'  target used by the claim    : S2-T1f (oracle placement AND k frozen) '
  f'{np.median(t1f):.4f}')
P(f'  quoted gap                                                 {np.median(a3)-np.median(t1f):.4f}')
P('')
P(f'  (i)  k-FREEZING, not placement    S2-T1 -> S2-T1f          '
  f'{np.median(t1)-np.median(t1f):.4f}  '
  f'({100*(np.median(t1)-np.median(t1f))/(np.median(a3)-np.median(t1f)):.0f}% of the gap)')
P(f'  (ii) ALREADY ACHIEVED by a shelved deployable arm')
P(f'       {best_dep[0]:38s} median init {np.median(bd):.4f}')
P(f'       S1-A3 -> {best_dep[0][:20]:20s}                  '
  f'{np.median(a3)-np.median(bd):.4f}  '
  f'({100*(np.median(a3)-np.median(bd))/(np.median(a3)-np.median(t1f)):.0f}% of the gap)')
P(f'  (iii) WHAT IS ACTUALLY LEFT       {best_dep[0][:20]:20s} -> S2-T1   '
  f'{np.median(bd)-np.median(t1):.4f}  '
  f'({100*(np.median(bd)-np.median(t1))/(np.median(a3)-np.median(t1f)):.0f}% of the gap)')
P('')
d = bd - t1
k = sum(1 for v in d if v < 0)
P(f'  paired {best_dep[0][:24]} vs S2-T1 (oracle placement, same machinery):')
P(f'    per-episode d(init_nrmse) = {[round(v,4) for v in d]}   med {np.median(d):+.4f}  '
  f'{k}/4  p={sign_p(k,4):.4f}')
P('')
P(f'  ratio, as quoted : {np.median(a3)/np.median(t1f):.2f}x')
P(f'  ratio, best deployable vs oracle placement WITHOUT the freeze : '
  f'{np.median(bd)/np.median(t1):.2f}x')

P('')
P('=' * 104)
P('E2 2x2 SEPARATION (coarse-grid step x absolute suppression radius), crowded 4')
P('=' * 104)
P(f'{"":16s} {"radius 0.200":>14s} {"radius 0.100":>14s}')
for gname, g in (('grid  81', ('U0', 'U3')), ('grid 401', ('U1', 'U2'))):
    P(f'{gname:16s} ' + ' '.join(
        f'{np.median([lv(e2, c, e, "init_nrmse") for e in CROWD]):14.4f}' for c in g))
P(f'  (anchor A0 = shipped: 81 grid, MIN_DK 0.15 in CELL units) '
  f'{np.median([lv(e2, "A0", e, "init_nrmse") for e in CROWD]):.4f}')
P('  -> both single-factor moves off A0 are available; the design is NOT additive')
P('     (U2 = both moves is worse than either alone).')

P('')
P('=' * 104)
P('SAME LADDER in final nRMSE')
P('=' * 104)
P(f'{"arm":32s} ' + ' '.join(f'{e.split("_")[1]+"_"+e.split("_")[2]:>13s}' for e in CROWD)
  + f' {"MEDIAN":>8s}')
tabn = {}
for name, rows, c, cls in ARMS:
    vals = [lv(rows, c, e, 'nRMSE') for e in CROWD]
    tabn[name] = vals
    P(f'{name:32s} ' + ' '.join(f'{v:13.4f}' for v in vals) + f' {np.median(vals):8.4f}')
letkf = 0.2394
P(f'  (frozen LETKF reference, s1s2 RESULT.md mean4 = {letkf:.4f})')

P('')
P('=' * 104)
P('HEADROOM AUDIT — why the 12-episode panel could not have detected these')
P('=' * 104)
li = {}
ALL12 = sorted({r['episode'] for r in e2})
for e in ALL12:
    li[e] = (lv(e2, 'A0', e, 'nRMSE'), lv(e2, 'U1', e, 'nRMSE'))
P(f'{"episode":42s} {"A0 nRMSE":>9s} {"max possible":>13s} {"U1 delta":>9s}')
for e in ALL12:
    a, b = li[e]
    P(f'{e:42s} {a:9.4f} {-a:13.4f} {b-a:+9.4f}')
noncrowd = [e for e in ALL12 if e not in CROWD]
P('')
P(f'  On the 8 non-crowded episodes the shipped arm is already at median nRMSE '
  f'{np.median([li[e][0] for e in noncrowd]):.4f}.')
P(f'  The largest delta ANY intervention could produce there is '
  f'{-np.median([li[e][0] for e in noncrowd]):.4f}, i.e. '
  f'{100*np.median([li[e][0] for e in noncrowd])/0.0974:.0f}% of the prize.')
P(f'  A 12-episode sign test in which 8 episodes are capped at |delta| <= 0.013 is')
P(f'  decided by the sign of noise in those 8, not by the effect in the 4 that can move.')
P(f'  U1 deltas, non-crowded 8: {[round(li[e][1]-li[e][0],4) for e in noncrowd]}')
P(f'  U1 deltas, crowded 4    : {[round(li[e][1]-li[e][0],4) for e in CROWD]}')

open(os.path.join(os.path.dirname(__file__), 'CON_rederive4.txt'), 'w').write('\n'.join(OUT) + '\n')
json.dump({'init_ladder': tab, 'nrmse_ladder': tabn},
          open(os.path.join(os.path.dirname(__file__), 'CON_rederive4.json'), 'w'), indent=1)
