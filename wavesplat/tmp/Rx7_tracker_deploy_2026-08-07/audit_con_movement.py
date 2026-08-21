"""CON audit — how much tracking actually happens at lr/1000, and where the
LR1000 win lives (router arm heterogeneity).  Read-only.
"""
import json, os
import statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(f'{HERE}/results.jsonl') if l.strip()]
rows = [r for r in rows if 'error' not in r]
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean']
        for n in json.load(open(f'{HERE}/nulls_rx7.json'))}
R = {(r['cell'], r['episode'], r['seed']): r for r in rows}
EPS = sorted({r['episode'] for r in rows})
SEEDS = sorted({r['seed'] for r in rows})


def med(c, e, k='nRMSE'):
    return st.median([R[(c, e, s)][k] for s in SEEDS])


KEPT = [e for e in EPS if med('AUTO', e) < NULL[e]]

print('=' * 78)
print('A. realised parameter movement: how much tracking is lr=5e-6 actually doing?')
print('=' * 78)
print(f'  {"cell":13s} {"kx bit-eq":>10s} {"rms|dw|/rms|w0|":>17s} {"rms|dphi| [rad]":>16s} '
      f'{"D_phi=rms(w0*dphi)":>19s}')
for c in ['AUTO-LR1000', 'AUTO-LR100', 'AUTO-LR10', 'AUTO']:
    eq = 0; tot = 0; dw = []; dp = []; Dp = []
    for e in EPS:
        for s in SEEDS:
            a = R[(c, e, s)]; b = R[('AUTO-NOTRACK', e, s)]
            tot += 1
            ka, kb = np.array(a['kx']), np.array(b['kx'])
            if ka.shape == kb.shape and np.array_equal(ka, kb):
                eq += 1
            else:
                continue
            w0 = np.array(b['w']); w1 = np.array(a['w'])
            p0 = np.array(b['phi']); p1 = np.array(a['phi'])
            d = (p1 - p0 + np.pi) % (2 * np.pi) - np.pi
            dw.append(np.sqrt(np.mean((w1 - w0) ** 2)) / max(np.sqrt(np.mean(w0 ** 2)), 1e-12))
            dp.append(np.sqrt(np.mean(d ** 2)))
            Dp.append(np.sqrt(np.mean((w0 * d) ** 2)))
    print(f'  {c:13s} {eq:6d}/{tot:<4d} {st.median(dw):17.6f} {st.median(dp):16.6f} '
          f'{st.median(Dp):19.6f}')
print('  (AUTO-NOTRACK is the frozen-init reference; TRACK_PHASE_PREDICT=False and')
print('   tracking_lr=0 make its update() a no-op apart from an idempotent steepness clamp.)')

print('\n' + '=' * 78)
print('B. router heterogeneity: AUTO is a ROUTER; where does the LR1000 win live?')
print('=' * 78)
arms = {}
for e in EPS:
    for s in SEEDS:
        arms.setdefault(R[('AUTO', e, s)]['auto_selected'], []).append((e, s))
for a, v in arms.items():
    print(f'  arm={a}: {len(v)} (ep,seed) over {len(set(x[0] for x in v))} episodes')
ep_arm = {}
for e in EPS:
    aa = {R[('AUTO', e, s)]['auto_selected'] for s in SEEDS}
    ep_arm[e] = 'mixed' if len(aa) > 1 else aa.pop()
for a in sorted(set(ep_arm.values())):
    ee = [e for e in KEPT if ep_arm[e] == a]
    if not ee:
        print(f'  {a}: 0 kept episodes')
        continue
    d = [med('AUTO-LR1000', e) - med('AUTO-NOTRACK', e) for e in ee]
    w = sum(1 for x in d if x < 0)
    print(f'  {a:10s} n={len(ee):2d}  medD(LR1000-floor)={st.median(d):+.6f}  win {w}/{len(ee)}')
    d2 = [med('AUTO', e) - med('AUTO-NOTRACK', e) for e in ee]
    print(f'             medD(AUTO-floor)   ={st.median(d2):+.6f}  '
          f'worse {sum(1 for x in d2 if x>0)}/{len(ee)}')

print('\n' + '=' * 78)
print('C. sea-state / component-count folds (n=4 each; PREREG bans "significant" here)')
print('=' * 78)
for key, sel in (('BF4', lambda e: 'BF4' in e), ('BF5', lambda e: 'BF5' in e),
                 ('15comp', lambda e: '15comp' in e), ('30comp', lambda e: '30comp' in e)):
    ee = [e for e in KEPT if sel(e)]
    d = [med('AUTO-LR1000', e) - med('AUTO-NOTRACK', e) for e in ee]
    do = [med('AUTO', e, 'avg_RMSE_observed') - med('AUTO-NOTRACK', e, 'avg_RMSE_observed')
          for e in ee]
    du = [med('AUTO', e, 'avg_RMSE_unobserved') - med('AUTO-NOTRACK', e, 'avg_RMSE_unobserved')
          for e in ee]
    print(f'  {key:7s} n={len(ee)}  LR1000 medD={st.median(d):+.6f} ({sum(1 for x in d if x<0)}/{len(ee)}) | '
          f'obs medD={st.median(do):+.4f} ({sum(1 for x in do if x<0)}/{len(ee)}) | '
          f'unobs medD={st.median(du):+.4f} ({sum(1 for x in du if x>0)}/{len(ee)} worse)')

print('\n' + '=' * 78)
print('D. monotone ladder — is the observed-region sign a mechanical function of lr?')
print('=' * 78)
LADDER = [('AUTO-NOTRACK', 0.0), ('AUTO-LR1000', 5e-6), ('AUTO-LR100', 5e-5),
          ('AUTO-LR10', 5e-4), ('AUTO', 5e-3)]
print(f'  {"lr":>9s} {"obs level":>10s} {"unobs level":>12s} {"nRMSE":>8s} '
      f'{"obs win vs floor":>17s} {"unobs worse vs floor":>21s}')
for c, lr in LADDER:
    lo = st.median([med(c, e, 'avg_RMSE_observed') for e in KEPT])
    lu = st.median([med(c, e, 'avg_RMSE_unobserved') for e in KEPT])
    lv = st.median([med(c, e) for e in KEPT])
    wo = sum(1 for e in KEPT if med(c, e, 'avg_RMSE_observed') <
             med('AUTO-NOTRACK', e, 'avg_RMSE_observed'))
    wu = sum(1 for e in KEPT if med(c, e, 'avg_RMSE_unobserved') >
             med('AUTO-NOTRACK', e, 'avg_RMSE_unobserved'))
    print(f'  {lr:9.0e} {lo:10.4f} {lu:12.4f} {lv:8.4f} {wo:12d}/{len(KEPT):<4d} '
          f'{wu:16d}/{len(KEPT):<4d}')
print('  -> 4/4 tracker-on cells improve the observed region 13/13, monotone in lr.')
print('     Unanimity in every arm at every strength is the signature of a mechanical')
print('     relation (gradient descent on the observation residual), not a stochastic finding.')
