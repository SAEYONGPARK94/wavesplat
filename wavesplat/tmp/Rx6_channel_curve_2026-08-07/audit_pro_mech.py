"""PRO mechanism + replication-split analysis.
(1) does the phase channel SATURATE (bounded D and bounded damage) while amplitude does not?
(2) split Rx-6 into the 16 episodes Rx-5 already used (indep2) and the 16 FRESH ones (indep3)
(3) how much of the system's flat-sea margin does drive give back?
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK'] + [f'{c}-S{i}' for c in 'WP' for i in range(1, 6)]
SLEV = ['S1', 'S2', 'S3', 'S4', 'S5']
NEEDED = {32: 22, 31: 22, 30: 21, 29: 21, 28: 20, 27: 20, 26: 19, 16: 13, 15: 12, 14: 12}


def sign_p(w, n):
    if n == 0:
        return 1.0
    hi = sum(math.comb(n, k) for k in range(w, n + 1)) / 2.0 ** n
    lo = sum(math.comb(n, k) for k in range(0, w + 1)) / 2.0 ** n
    return min(1.0, 2 * min(hi, lo))


def med(x):
    return float(np.median(np.asarray(x, float)))


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


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


def test(d, label, need=None):
    nz = [v for v in d if v != 0]
    w = sum(1 for v in nz if v > 0)
    n = len(nz)
    need = need if need else NEEDED.get(n, n)
    tag = 'PASS' if (w >= need or n - w >= need) else 'null'
    print(f'{label:56s} med {med(d):>+9.5f}  {w:>3d}/{n:<3d} p={sign_p(w, n):<9.6f} {tag}')


print('=' * 100)
print('E. SATURATION: is the phase channel BOUNDED and the amplitude channel NOT?')
print('=' * 100)
rmsw = med([med([float(np.sqrt(np.mean(np.array(raw['AUTO-NOTRACK'][(e, s)]['w'], float) ** 2)))
                 for s in seeds]) for e in eps_all])
print(f'rms(w) at the floor (Phase-1 params)      = {rmsw:.6f} m')
print(f'ceiling for D_phi if phase fully randomises = rms(w) * pi/sqrt(3) = {rmsw * math.pi / math.sqrt(3):.6f} m')
print('  (wrap(dphi) in (-pi,pi]; uniform dphi has rms pi/sqrt(3)=1.8138 rad -> D_phi has a HARD CEILING)')
print('  amplitude has NO such ceiling: |dw| is unbounded above.\n')
print(f'{"cell":8s} {"lr":>12s} {"rms|dphi|[rad]":>15s} {"D_phi[m]":>10s} {"% of ceiling":>13s} {"damage":>9s}')
for S in SLEV:
    c = 'P-' + S
    lr = raw[c][(eps_all[0], seeds[0])]['cfg']['tracking_lr']
    rp = med([med([float(np.sqrt(np.mean(wrap(np.array(raw[c][(e, s)]['phi'], float)
                                             - np.array(raw['AUTO-NOTRACK'][(e, s)]['phi'], float)) ** 2)))
                   for s in seeds]) for e in eps_all])
    dp = med([med([float(np.sqrt(np.mean((np.array(raw['AUTO-NOTRACK'][(e, s)]['w'], float)
                                          * wrap(np.array(raw[c][(e, s)]['phi'], float)
                                                 - np.array(raw['AUTO-NOTRACK'][(e, s)]['phi'], float))) ** 2)))
                   for s in seeds]) for e in eps_all])
    dmg = med([level[c][e] - level['AUTO-NOTRACK'][e] for e in EPS])
    print(f'{c:8s} {lr:>12.3e} {rp:>15.4f} {dp:>10.5f} {dp / (rmsw * math.pi / math.sqrt(3)) * 100:>12.1f}% {dmg:>+9.5f}')
print()
print(f'{"cell":8s} {"lr":>12s} {"rms|dw|[m]":>15s} {"D_w/rms(w)":>10s} {"":>13s} {"damage":>9s}')
for S in SLEV:
    c = 'W-' + S
    lr = raw[c][(eps_all[0], seeds[0])]['cfg']['tracking_lr']
    dw = med([med([float(np.sqrt(np.mean((np.array(raw[c][(e, s)]['w'], float)
                                          - np.array(raw['AUTO-NOTRACK'][(e, s)]['w'], float)) ** 2)))
                   for s in seeds]) for e in eps_all])
    dmg = med([level[c][e] - level['AUTO-NOTRACK'][e] for e in EPS])
    print(f'{c:8s} {lr:>12.3e} {dw:>15.5f} {dw / rmsw:>10.3f} {"":>13s} {dmg:>+9.5f}')

print('\nE2. within-channel plateau test (1 knob = tracking_lr, PREREG-legal contrast):')
for a, b in [('P-S1', 'P-S2'), ('P-S1', 'P-S3'), ('P-S2', 'P-S3'),
             ('W-S1', 'W-S2'), ('W-S1', 'W-S3'), ('W-S2', 'W-S3')]:
    lra = raw[a][(eps_all[0], seeds[0])]['cfg']['tracking_lr']
    lrb = raw[b][(eps_all[0], seeds[0])]['cfg']['tracking_lr']
    test([level[a][e] - level[b][e] for e in EPS], f'  {a} - {b}  (lr x{lra / lrb:.1f})')

print('\n' + '=' * 100)
print('F. REPLICATION SPLIT: Rx-6 contains the 16 episodes Rx-5 already scored')
print('=' * 100)
OLD = [e for e in eps_all if any(f'seed{i}_' in e for i in range(4))]     # indep2 = Rx-5 set
NEW = [e for e in eps_all if any(f'seed{i}_' in e for i in range(4, 8))]  # indep3 = FRESH
print(f'  indep2 (already used by Rx-5, fresh estimator seeds only): {len(OLD)} episodes')
print(f'  indep3 (FRESH episodes, never scored before):              {len(NEW)} episodes')
for name, sub in [('indep2 (Rx-5 reuse)', OLD), ('indep3 (FRESH)', NEW), ('all 32', eps_all)]:
    s = [e for e in sub if e in EPS]
    test([level['AUTO-NOTRACK'][e] - level['AUTO'][e] for e in s],
         f'  AUTO-NOTRACK - AUTO  [{name}] n={len(s)}')
print('\n  C(S1) on the same split:')
for name, sub in [('indep2 (Rx-5 reuse)', OLD), ('indep3 (FRESH)', NEW)]:
    s = [e for e in sub if e in EPS]
    test([level['W-S1'][e] - level['P-S1'][e] for e in s], f'    C(S1) [{name}] n={len(s)}')
print('\n  post-hoc D-matched channel offset on the same split:')
for name, sub in [('indep2 (Rx-5 reuse)', OLD), ('indep3 (FRESH)', NEW), ('all 31', EPS)]:
    s = [e for e in sub if e in EPS]
    test([level['W-S1'][e] - level['P-S2'][e] for e in s], f'    W-S1 - P-S2 [{name}] n={len(s)}')

print('\n' + '=' * 100)
print('G. HOW MUCH OF THE FLAT-SEA MARGIN DOES THE TRACKER GIVE BACK?')
print('=' * 100)
print('  per-episode: margin(floor) = null - AUTO-NOTRACK ; given-back = cell - AUTO-NOTRACK')
print(f'{"cell":14s} {"med margin":>11s} {"med givenback":>14s} {"med per-ep FRACTION given back":>32s}')
for c in ['AUTO', 'W-S1', 'P-S1', 'W-S2', 'P-S2', 'W-S3', 'P-S3', 'W-S4', 'P-S4']:
    marg = [nulls[e] - level['AUTO-NOTRACK'][e] for e in EPS]
    gb = [level[c][e] - level['AUTO-NOTRACK'][e] for e in EPS]
    frac = [g / m for g, m in zip(gb, marg)]
    print(f'{c:14s} {med(marg):>11.5f} {med(gb):>14.5f} {med(frac) * 100:>31.1f}%')

print('\nG2. spec-level diagnostics on the tracker-off contrast (descriptive):')
for k in ['amp_rel_error_mean', 'direction_error_deg_mean', 'k_mag_rel_error_mean',
          'omega_rel_error_mean', 'detection_rate']:
    per = {c: {e: med([raw[c][(e, s)]['spec'][k] for s in seeds]) for e in EPS}
           for c in ['AUTO', 'AUTO-NOTRACK', 'W-S1', 'P-S1']}
    d = [per['AUTO-NOTRACK'][e] - per['AUTO'][e] for e in EPS]
    w = sum(1 for v in d if v > 0)
    print(f'  {k:26s} AUTO {med(list(per["AUTO"].values())):8.4f}  NOTRACK '
          f'{med(list(per["AUTO-NOTRACK"].values())):8.4f}  W-S1 {med(list(per["W-S1"].values())):8.4f}  '
          f'P-S1 {med(list(per["P-S1"].values())):8.4f}  |  paired {med(d):+.4f} {w}/{len(d)} p={sign_p(w, len(d)):.4f}')

print('\nG3. does the tracker hurt because it AMPLIFIES?  (m_mean = frame-wise vs flat sea)')
for c in ['AUTO-NOTRACK', 'W-S5', 'W-S4', 'W-S3', 'W-S2', 'W-S1',
          'P-S5', 'P-S4', 'P-S3', 'P-S2', 'P-S1', 'AUTO']:
    vr = med([med([raw[c][(e, s)]['Variance_Ratio'] for s in seeds]) for e in EPS])
    sh = med([med([raw[c][(e, s)]['shrink_mean'] for s in seeds]) for e in EPS])
    dc = med([med([raw[c][(e, s)]['decorr_mean'] for s in seeds]) for e in EPS])
    mm = med([med([raw[c][(e, s)]['m_mean'] for s in seeds]) for e in EPS])
    rho = med([med([raw[c][(e, s)]['Pearson_rho'] for s in seeds]) for e in EPS])
    print(f'  {c:14s} VR {vr:6.3f}  rho {rho:6.4f}  m {mm:6.4f}  shrink {sh:7.4f}  decorr {dc:7.4f}')
