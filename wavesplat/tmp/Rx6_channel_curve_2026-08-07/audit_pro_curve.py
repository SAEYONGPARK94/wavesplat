"""PRO deep analysis: (a) is damage ONE function of realized displacement D?
(b) is C(S4) an on/off contrast in disguise (Rx-5 pathology)?  (c) how much drive
pushes the tracker past the flat-sea null?  (d) power done right.
All judgment quantities are per-episode PAIRED; no ratio-of-aggregates, no
difference-of-block-medians, no absolute-value rule.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK'] + [f'{c}-S{i}' for c in 'WP' for i in range(1, 6)]
SLEV = ['S1', 'S2', 'S3', 'S4', 'S5']
NEEDED = {32: 22, 31: 22, 30: 21, 29: 21, 28: 20, 27: 20, 26: 19, 25: 18, 24: 18,
          23: 17, 22: 17, 21: 16, 20: 15, 19: 15, 18: 14, 17: 13, 16: 13, 15: 12,
          14: 12, 13: 11, 12: 10, 11: 10, 10: 9, 9: 8, 8: 8, 7: 7}


def sign_p(w, n):
    if n == 0:
        return 1.0
    hi = sum(math.comb(n, k) for k in range(w, n + 1)) / 2.0 ** n
    lo = sum(math.comb(n, k) for k in range(0, w + 1)) / 2.0 ** n
    return min(1.0, 2 * min(hi, lo))


def med(x):
    return float(np.median(np.asarray(x, float)))


raw = defaultdict(dict)
for line in open(os.path.join(HERE, 'results.jsonl')):
    r = json.loads(line)
    raw[r['cell']][(r['episode'], r['seed'])] = r
eps_all = sorted({k[0] for k in raw['AUTO']})
seeds = sorted({k[1] for k in raw['AUTO']})
nulls = {r['episode']: r['zero_null_tracking_nRMSE_mean']
         for r in json.load(open(os.path.join(HERE, 'nulls_rx6.json')))}

level = {c: {e: med([raw[c][(e, s)]['nRMSE'] for s in seeds]) for e in eps_all} for c in CELLS}
EPS = [e for e in eps_all if level['AUTO'][e] < nulls[e]]
N = len(EPS)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


Dnat = {}     # cell -> ep -> channel-native realized field displacement [m]
for c in CELLS:
    Dnat[c] = {}
    for e in eps_all:
        v = []
        for s in seeds:
            b, u = raw['AUTO-NOTRACK'][(e, s)], raw[c][(e, s)]
            w0 = np.array(b['w'], float)
            p0 = np.array(b['phi'], float)
            if c.startswith('W') or c == 'AUTO-NOTRACK':
                v.append(float(np.sqrt(np.mean((np.array(u['w'], float) - w0) ** 2))))
            elif c.startswith('P'):
                v.append(float(np.sqrt(np.mean((w0 * wrap(np.array(u['phi'], float) - p0)) ** 2))))
            else:
                dw = np.array(u['w'], float) - w0
                dp = w0 * wrap(np.array(u['phi'], float) - p0)
                v.append(float(np.sqrt(np.mean(dw ** 2 + dp ** 2))))
        Dnat[c][e] = med(v)

dmg = {c: {e: level[c][e] - level['AUTO-NOTRACK'][e] for e in eps_all} for c in CELLS}


def test(dvals, label, need=None):
    nz = [v for v in dvals if v != 0]
    w = sum(1 for v in nz if v > 0)
    n = len(nz)
    need = need or NEEDED.get(n, n)
    tag = 'PASS' if (w >= need or n - w >= need) else 'null'
    print(f'{label:52s} med {med(dvals):>+9.5f}  {w:>3d}/{n:<3d} p={sign_p(w, n):<9.5f} {tag}')
    return med(dvals), w, n, sign_p(w, n)


print('=' * 100)
print(f'A. IS DAMAGE ONE FUNCTION OF REALIZED DISPLACEMENT D?   (PRIMARY set n={N})')
print('=' * 100)
print('\nA1. cross-channel cells matched on D (POST-HOC pairing, exploratory, not the frozen PRIMARY):')
print(f'{"pair":22s} {"D_W":>9s} {"D_P":>9s} {"D mismatch":>11s}   paired damage difference')
CAND = [('W-S1', 'P-S2'), ('W-S2', 'P-S3'), ('W-S3', 'P-S4'), ('W-S1', 'P-S1'),
        ('W-S2', 'P-S2'), ('W-S4', 'P-S5'), ('W-S5', 'P-S5')]
for a, b in CAND:
    da, db = med([Dnat[a][e] for e in EPS]), med([Dnat[b][e] for e in EPS])
    print(f'{a + " vs " + b:22s} {da:>9.5f} {db:>9.5f} {db / da:>10.3f}x   ', end='')
    test([level[a][e] - level[b][e] for e in EPS], '')

print('\nA2. per-episode D-matched: for each episode pick the P cell whose D is closest to each W cell')
for wc in ['W-S1', 'W-S2', 'W-S3', 'W-S4']:
    diffs, mism, picks = [], [], defaultdict(int)
    for e in EPS:
        tgt = Dnat[wc][e]
        pc = min(('P-' + S for S in SLEV), key=lambda c: abs(math.log(max(Dnat[c][e], 1e-12) / max(tgt, 1e-12))))
        picks[pc] += 1
        diffs.append(dmg[wc][e] - dmg[pc][e])
        mism.append(Dnat[pc][e] / max(tgt, 1e-12))
    print(f'  {wc}: median D mismatch {med(mism):.3f}x  picks {dict(picks)}')
    test(diffs, f'    {wc} - (per-episode D-nearest P cell)')

print('\nA3. does D alone order the damage?  (10 cells, PRIMARY set)')
pts = [(med([Dnat[c][e] for e in EPS]), med([dmg[c][e] for e in EPS]), c)
       for c in CELLS if c[0] in 'WP']
pts.sort()
for d, m, c in pts:
    print(f'   {c:6s} D={d:.5f}  damage={m:+.5f}')
# monotone-in-D violations
viol = [(pts[i][2], pts[i + 1][2]) for i in range(len(pts) - 1) if pts[i + 1][1] < pts[i][1]]
print(f'   monotonicity violations along D: {viol}')

# leave-one-channel-out interpolation: predict W damage from the P curve and vice versa
def interp_curve(src, tgt):
    xs = np.log10([med([Dnat['%s-%s' % (src, S)][e] for e in EPS]) for S in SLEV])
    ys = np.array([med([dmg['%s-%s' % (src, S)][e] for e in EPS]) for S in SLEV])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    rows = []
    for S in SLEV:
        x = math.log10(med([Dnat['%s-%s' % (tgt, S)][e] for e in EPS]))
        if x < xs[0] or x > xs[-1]:
            rows.append((S, None, None))
            continue
        pred = float(np.interp(x, xs, ys))
        obs = med([dmg['%s-%s' % (tgt, S)][e] for e in EPS])
        rows.append((S, pred, obs))
    return rows


print('\nA4. predict one channel from the OTHER channel curve (log10 D interpolation, in-range only):')
for src, tgt in [('P', 'W'), ('W', 'P')]:
    print(f'   {src}-curve -> predict {tgt}:')
    for S, pred, obs in interp_curve(src, tgt):
        if pred is None:
            print(f'      {tgt}-{S}: out of {src} D range, no prediction')
        else:
            print(f'      {tgt}-{S}: predicted {pred:+.5f}  observed {obs:+.5f}  residual {obs - pred:+.5f}')

print('\n' + '=' * 100)
print('B. IS C(S4) AN ON/OFF CONTRAST IN DISGUISE?  (Rx-5 PRIMARY-A pathology check)')
print('=' * 100)
for S in SLEV:
    real = [level['W-' + S][e] - level['P-' + S][e] for e in EPS]
    surr = [level['AUTO-NOTRACK'][e] - level['P-' + S][e] for e in EPS]
    r = float(np.corrcoef(real, surr)[0, 1])
    rw = sum(1 for v in real if v > 0)
    sw = sum(1 for v in surr if v > 0)
    print(f'  C({S}) real  {med(real):+.5f} {rw}/{len(real)}  |  floor-surrogate {med(surr):+.5f} '
          f'{sw}/{len(surr)}  |  per-episode r = {r:+.4f}  |  point-estimate gap '
          f'{abs(med(real) - med(surr)):.5f}')
print('  (Rx-5 called PRIMARY-A void at r=0.9921 with a 2.8% point gap)')

print('\nB2. degeneracy ledger: is each cell distinguishable from BOTH endpoints? (PREREG sec.5)')
print(f'{"cell":8s} {"vs AUTO":>28s} {"vs FLOOR":>28s}   status')
for c in [f'{ch}-S{i}' for ch in 'WP' for i in range(1, 6)]:
    da = [level[c][e] - level['AUTO'][e] for e in EPS]
    df = [level[c][e] - level['AUTO-NOTRACK'][e] for e in EPS]
    wa = sum(1 for v in da if v > 0)
    wf = sum(1 for v in df if v > 0)
    pa, pf = sign_p(wa, len(da)), sign_p(wf, len(df))
    oka = wa >= NEEDED[N] or len(da) - wa >= NEEDED[N]
    okf = wf >= NEEDED[N] or len(df) - wf >= NEEDED[N]
    st = 'OK' if oka and okf else ('== FLOOR (degenerate)' if not okf else '== AUTO (degenerate)')
    print(f'{c:8s} {med(da):>+9.5f} {wa:>3d}/{N} p={pa:<7.4f} {med(df):>+9.5f} {wf:>3d}/{N} p={pf:<7.4f}   {st}')

print('\n' + '=' * 100)
print('C. HOW MUCH DRIVE PUSHES THE TRACKER PAST THE FLAT-SEA NULL?')
print('=' * 100)
print(f'{"cell":14s} {"D[m]":>9s} {"beats null":>11s} {"m>1":>6s} {"med(cell-null)":>15s} '
      f'{"eps LOSING to flat sea":>23s}')
rows = []
for c in ['AUTO-NOTRACK', 'W-S5', 'P-S5', 'W-S4', 'P-S4', 'W-S3', 'P-S3',
          'W-S2', 'P-S2', 'W-S1', 'P-S1', 'AUTO']:
    d = med([Dnat[c][e] for e in eps_all])
    nb = sum(1 for e in eps_all if level[c][e] < nulls[e])
    mg = sum(1 for e in eps_all if med([raw[c][(e, s)]['m_mean'] for s in seeds]) > 1.0)
    marg = med([level[c][e] - nulls[e] for e in eps_all])
    lose = [e[8:-3] for e in eps_all if level[c][e] >= nulls[e]]
    rows.append((c, d, nb, mg, marg))
    print(f'{c:14s} {d:>9.5f} {nb:>8d}/32 {mg:>4d}/32 {marg:>+15.5f} {len(lose):>20d}')
print('\n  correlation across those 12 cells: D vs #episodes losing to flat sea =', end=' ')
xs = np.array([r[1] for r in rows])
ys = np.array([32 - r[2] for r in rows])
print(f'{np.corrcoef(xs, ys)[0, 1]:+.4f} (Pearson), '
      f'{np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1]:+.4f} (Spearman)')
print('  margin to the null (negative = better than flat sea), paired sign test vs 0:')
for c in ['AUTO-NOTRACK', 'W-S3', 'W-S2', 'W-S1', 'P-S1', 'AUTO']:
    test([level[c][e] - nulls[e] for e in eps_all], f'    {c} - null (all 32)')

print('\n' + '=' * 100)
print('D. TRACKER-OFF CONTRAST, POWER DONE RIGHT')
print('=' * 100)
d_off = [level['AUTO-NOTRACK'][e] - level['AUTO'][e] for e in EPS]
mo, wo, no, po = test(d_off, 'AUTO-NOTRACK - AUTO (PRIMARY n=31)')
print(f'  wins in the "off is better" direction: {no - wo}/{no}')
d_off32 = [level['AUTO-NOTRACK'][e] - level['AUTO'][e] for e in eps_all]
test(d_off32, 'AUTO-NOTRACK - AUTO (all 32, no null exclusion)')

rng = np.random.default_rng(20260807)


def power_and_mde(d, need, target=0.80, nboot=6000):
    d = np.asarray(d, float)
    n = len(d)
    samp = rng.choice(d, size=(nboot, n), replace=True)
    w = (samp > 0).sum(axis=1)
    obs_pw = float(((w >= need) | (n - w >= need)).mean())
    d0 = d - np.median(d)
    lo, hi = 0.0, 0.3
    for _ in range(40):
        mid = (lo + hi) / 2
        s2 = rng.choice(d0, size=(nboot, n), replace=True) + mid
        w2 = (s2 > 0).sum(axis=1)
        if float(((w2 >= need) | (n - w2 >= need)).mean()) < target:
            lo = mid
        else:
            hi = mid
    return obs_pw, hi


headroom = abs(mo)
pw, m = power_and_mde(d_off, NEEDED[N])
print(f'  headroom |AUTO-NOTRACK - AUTO| = {headroom:.5f}')
print(f'  observed power {pw:.3f}   MDE(80%) {m:.5f} = {m / headroom * 100:.1f}% of headroom  '
      f'obs/MDE {headroom / m:.2f}x')
fails = 0
for i in range(len(d_off)):
    dd = [v for j, v in enumerate(d_off) if j != i]
    w = sum(1 for v in dd if v > 0)
    if not (w >= NEEDED[len(dd)] or len(dd) - w >= NEEDED[len(dd)]):
        fails += 1
print(f'  LOO (n=31 -> 30, needed {NEEDED[30]}): threshold failures {fails}/{len(d_off)}')
print(f'  size structure: favourable sum {sum(v for v in d_off if v < 0):+.5f} '
      f'vs adverse sum {sum(v for v in d_off if v > 0):+.5f}  '
      f'ratio {abs(sum(v for v in d_off if v < 0)) / max(sum(v for v in d_off if v > 0), 1e-9):.1f}x')
print('  seed-wise stability (each seed scored alone, n=31):')
for s in seeds:
    dd = [raw['AUTO-NOTRACK'][(e, s)]['nRMSE'] - raw['AUTO'][(e, s)]['nRMSE'] for e in EPS]
    w = sum(1 for v in dd if v > 0)
    print(f'    seed {s}: med {med(dd):+.5f}  off-better {len(dd) - w}/{len(dd)}  p={sign_p(w, len(dd)):.5f}')

print('\n  C(S) power (PRIMARY quantities):')
for S in SLEV:
    d = [level['W-' + S][e] - level['P-' + S][e] for e in EPS]
    pw, m = power_and_mde(d, NEEDED[N])
    print(f'    C({S}): obs {med(d):+.5f}  power {pw:.3f}  MDE(80%) {m:.5f} = '
          f'{m / headroom * 100:5.1f}% of headroom')
