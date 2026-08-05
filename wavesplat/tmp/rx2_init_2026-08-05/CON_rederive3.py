"""CON part 3 — the crowded-subset re-read of every candidate that was declared dead.

The elimination argument behind "initialisation is all that is left" mixed two panels:
the prize is quoted on 4 CROWDED episodes, but grid / suppression-radius / min-separation
were killed on the 12-episode paper panel, 8 of whose episodes have nRMSE ~= 0.013 and
cannot move by the size of the prize even in principle.

This script re-reads every E1 / E2 cell on the SAME 4 crowded episodes the prize is quoted
on, for both nRMSE and init_nrmse, with a seed bootstrap.

EXPLORATORY, POST-HOC STRATIFICATION. n=4, two-sided sign-test floor p=0.125, alpha=0.05
unreachable. Nothing here is a mechanism claim; it is a scope-of-the-null claim.
"""
import json, os
import numpy as np
from math import comb

TMP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


def load(path, key):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        r['_cell'] = r[key]; rows.append(r)
    return rows


def sign_p(k, n):
    if n == 0:
        return float('nan')
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def raw(rows, metric):
    d = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        d.setdefault((r['_cell'], r['episode']), []).append(float(v))
    return d


def contrast(rows, a, b, metric, eps, nboot=8000, seed=3):
    d = raw(rows, metric)
    E = [e for e in eps if (a, e) in d and (b, e) in d]
    if not E:
        return None
    dl = [float(np.median(d[(b, e)]) - np.median(d[(a, e)])) for e in E]
    k = sum(1 for v in dl if v < 0)
    rng = np.random.default_rng(seed)
    bs = np.empty(nboot)
    for i in range(nboot):
        bs[i] = np.median([np.median(rng.choice(d[(b, e)], len(d[(b, e)])))
                           - np.median(rng.choice(d[(a, e)], len(d[(a, e)])))
                           for e in E])
    return dict(n=len(E), med=float(np.median(dl)), k=k, p=sign_p(k, len(E)),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                deltas=dl, eps=E)


CROWD = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
         'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
e2 = load(os.path.join(TMP, 'e2_grid_sep_2026-08-05', 'results.jsonl'), 'cell')
e1 = load(os.path.join(TMP, 'e1_grid_2026-08-05', 'results.jsonl'), 'cell')
s1s2 = load(os.path.join(TMP, 's1s2_2026-08-05', 'results.jsonl'), 'arm')
ALL12 = sorted({r['episode'] for r in e2})

P('=' * 100)
P('REPLICATION CHECK — E2 cell A0 vs s1s2 arm S1-A0 (identical config, separate runs)')
P('=' * 100)
d_e2 = raw(e2, 'nRMSE'); d_s = raw(s1s2, 'nRMSE')
for e in CROWD:
    a = np.median(d_e2[('A0', e)]) if ('A0', e) in d_e2 else float('nan')
    b = np.median(d_s[('S1-A0', e)]) if ('S1-A0', e) in d_s else float('nan')
    P(f'  {e:42s} E2-A0 {a:.4f}   s1s2-S1-A0 {b:.4f}   diff {a-b:+.4f}')
P('')

for tag, rows, base, cells in [
        ('E2 (grid / suppression radius / min-sep)', e2, 'A0',
         ['U0', 'U1', 'U2', 'U1M2', 'U1M5']),
        ('E1 (placement / suppression)', e1, 'A', ['B', 'C', 'D', 'E'])]:
    P('=' * 100)
    P(f'{tag}: SAME cells, two panels')
    P('=' * 100)
    P(f'{"cell":6s} {"panel":10s} {"n":>3s} {"med dnRMSE":>11s} {"k/n":>6s} {"p":>7s} '
      f'{"95% CI(seed)":>21s} | {"med dinit":>10s} {"k/n":>6s}')
    for c in cells:
        for pname, eps in (('12-ep', ALL12), ('crowded4', CROWD)):
            r = contrast(rows, base, c, 'nRMSE', eps)
            ri = contrast(rows, base, c, 'init_nrmse', eps)
            if r is None:
                continue
            P(f'{c:6s} {pname:10s} {r["n"]:3d} {r["med"]:+11.4f} {r["k"]:3d}/{r["n"]:<2d} '
              f'{r["p"]:7.4f} [{r["ci"][0]:+7.4f},{r["ci"][1]:+7.4f}] | '
              f'{ri["med"]:+10.4f} {ri["k"]:3d}/{ri["n"]:<2d}')
        P('')

P('=' * 100)
P('THE POINT')
P('=' * 100)
r = contrast(e2, 'A0', 'U1', 'nRMSE', CROWD)
dl_prize = None
dp = raw(s1s2, 'nRMSE')
dl_prize = [float(np.median(dp[('S2-T1f', e)]) - np.median(dp[('S1-A0', e)]))
            for e in CROWD]
P(f'  quoted initialisation prize (crowded 4)  : {np.median(dl_prize):+.4f}')
P(f'  grid 81->401, SAME crowded 4             : {r["med"]:+.4f}  '
  f'{r["k"]}/{r["n"]}  p={r["p"]:.4f}  = {abs(r["med"]/np.median(dl_prize)):.0%} of the prize')
P(f'    per-episode: {[round(v,4) for v in r["deltas"]]}')
P(f'  ...and this candidate is on the "already dead" list because on the 12-episode panel')
P(f'  it reads 6/12, p=1.000.')

json.dump({'crowd_recheck': {f'{c}': contrast(e2, 'A0', c, 'nRMSE', CROWD)
                             for c in ['U0', 'U1', 'U2', 'U1M2', 'U1M5']},
           'e1_crowd': {f'{c}': contrast(e1, 'A', c, 'nRMSE', CROWD)
                        for c in ['B', 'C', 'D', 'E']}},
          open(os.path.join(os.path.dirname(__file__), 'CON_rederive3.json'), 'w'),
          indent=1, default=str)
open(os.path.join(os.path.dirname(__file__), 'CON_rederive3.txt'), 'w').write('\n'.join(OUT) + '\n')
