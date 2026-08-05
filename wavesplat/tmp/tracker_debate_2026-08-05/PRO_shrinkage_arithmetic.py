"""PRO agent, proposition G. Decompose the tracker's damage into
(a) correlation loss and (b) variance miscalibration, using the closed identity

    nRMSE = RMSE/Hs,  Hs = 4*std(gt),  so with zero-mean fields
    nRMSE = sqrt(1 + v - 2*rho*sqrt(v)) / 4,      v = Variance_Ratio

which is minimised at v = rho^2 with value sqrt(1-rho^2)/4.

No new wave runs: reads tmp/s1s2_2026-08-05/results.jsonl only.
"""
import json
import os
import numpy as np
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 's1s2_2026-08-05', 'results.jsonl')

EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SHORT = {e: e.split('_')[1] + '_' + e.split('_')[2] for e in EPS}
ARMS = ['S1-A0', 'X-A0-notrack', 'S2-T1f', 'X-T1f-notrack', 'S2-C1', 'S2-T1',
        'S1-A1', 'S1-A2', 'S1-A3']
FIELDS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Hs_error',
          'avg_RMSE_unobserved', 'init_nrmse']

rec = {}
for line in open(SRC):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'error' in r:
        continue
    rec.setdefault((r['arm'], r['episode']), {})[r['seed']] = r


def med(arm, ep, f):
    d = rec.get((arm, ep))
    if not d:
        return np.nan
    return float(np.median([v[f] for v in d.values()]))


def nrmse_from(rho, v):
    """closed-form nRMSE implied by (rho, VR) under zero-mean fields"""
    return np.sqrt(max(1.0 + v - 2.0 * rho * np.sqrt(v), 0.0)) / 4.0


def nrmse_opt(rho):
    """nRMSE after the BEST possible scalar rescaling of the estimate (v -> rho^2)"""
    return np.sqrt(max(1.0 - rho ** 2, 0.0)) / 4.0


print('=' * 100)
print('TABLE 1 -- identity check: does nRMSE = sqrt(1+v-2*rho*sqrt(v))/4 reproduce the measured nRMSE?')
print('=' * 100)
print(f"{'arm':<15}{'episode':<12}{'rho':>8}{'VR':>8}{'nRMSE_meas':>12}"
      f"{'nRMSE_pred':>12}{'rel.err':>9}")
rows = {}
for arm in ARMS:
    for ep in EPS:
        rho = med(arm, ep, 'Pearson_rho')
        v = med(arm, ep, 'Variance_Ratio')
        n = med(arm, ep, 'nRMSE')
        if np.isnan(rho):
            continue
        p = nrmse_from(rho, v)
        rows[(arm, ep)] = dict(rho=rho, v=v, n=n, pred=p, opt=nrmse_opt(rho))
        print(f'{arm:<15}{SHORT[ep]:<12}{rho:8.3f}{v:8.3f}{n:12.4f}{p:12.4f}'
              f'{(p - n) / n * 100:8.1f}%')

print()
print('=' * 100)
print('TABLE 2 -- OPTIMAL-SHRINKAGE COUNTERFACTUAL (per episode, median-of-3)')
print('  nRMSE_opt = sqrt(1-rho^2)/4  = the score after the best possible scalar rescale')
print('  recovered = nRMSE_meas - nRMSE_opt   (how much a perfect variance fix buys)')
print('=' * 100)
print(f"{'arm':<15}{'episode':<12}{'rho':>7}{'VR':>7}{'VR_opt':>8}"
      f"{'nRMSE':>9}{'nRMSE_opt':>11}{'recovered':>11}{'%recov':>8}")
for arm in ['S1-A0', 'X-A0-notrack', 'S2-T1f', 'X-T1f-notrack']:
    for ep in EPS:
        r = rows[(arm, ep)]
        rc = r['n'] - r['opt']
        print(f"{arm:<15}{SHORT[ep]:<12}{r['rho']:7.3f}{r['v']:7.3f}"
              f"{r['rho']**2:8.3f}{r['n']:9.4f}{r['opt']:11.4f}{rc:11.4f}"
              f"{rc / r['n'] * 100:7.1f}%")
    print('-' * 100)

print()
print('=' * 100)
print('TABLE 3 -- ARM MEANS over the 4 crowded episodes')
print('=' * 100)
print(f"{'arm':<15}{'rho':>8}{'VR':>8}{'VR_opt':>8}{'nRMSE':>9}{'nRMSE_opt':>11}"
      f"{'over/under-shrunk':>20}")
means = {}
for arm in ['S1-A0', 'X-A0-notrack', 'S2-T1f', 'X-T1f-notrack', 'S2-C1', 'S2-T1']:
    rr = [rows[(arm, e)] for e in EPS]
    m = dict(rho=np.mean([x['rho'] for x in rr]), v=np.mean([x['v'] for x in rr]),
             n=np.mean([x['n'] for x in rr]), opt=np.mean([x['opt'] for x in rr]))
    means[arm] = m
    tag = 'OVER-shrunk' if m['v'] < m['rho'] ** 2 else 'under-shrunk'
    print(f"{arm:<15}{m['rho']:8.3f}{m['v']:8.3f}{m['rho']**2:8.3f}{m['n']:9.4f}"
          f"{m['opt']:11.4f}{tag:>20}")

print()
print('=' * 100)
print('TABLE 4 -- PAIRED tracker-ON minus tracker-OFF, per episode (median-of-3)')
print('  positive nRMSE delta = tracking HURTS.  negative rho delta = tracking HURTS.')
print('=' * 100)
for on, off, tag in [('S1-A0', 'X-A0-notrack', 'beamform placement'),
                     ('S2-T1f', 'X-T1f-notrack', 'oracle placement')]:
    print(f'--- {tag}: {on}  vs  {off} ---')
    print(f"{'episode':<12}{'d_nRMSE':>10}{'d_rho':>9}{'d_VR':>9}{'d_Hs_err':>10}"
          f"{'d_RMSEunobs':>13}{'d_nRMSE_opt':>13}")
    dn, dr, dv, dopt, dh, du = [], [], [], [], [], []
    for ep in EPS:
        a, b = rows[(on, ep)], rows[(off, ep)]
        x = a['n'] - b['n']; y = a['rho'] - b['rho']; z = a['v'] - b['v']
        o = a['opt'] - b['opt']
        h = med(on, ep, 'Hs_error') - med(off, ep, 'Hs_error')
        u = med(on, ep, 'avg_RMSE_unobserved') - med(off, ep, 'avg_RMSE_unobserved')
        dn.append(x); dr.append(y); dv.append(z); dopt.append(o); dh.append(h); du.append(u)
        print(f'{SHORT[ep]:<12}{x:+10.4f}{y:+9.3f}{z:+9.3f}{h:+10.3f}{u:+13.4f}{o:+13.4f}')
    def s(a):
        return f'{sum(1 for q in a if q > 0)}/{len(a)}+'
    print(f"{'median':<12}{np.median(dn):+10.4f}{np.median(dr):+9.3f}"
          f"{np.median(dv):+9.3f}{np.median(dh):+10.3f}{np.median(du):+13.4f}"
          f"{np.median(dopt):+13.4f}")
    print(f"{'sign(+)':<12}{s(dn):>10}{s(dr):>9}{s(dv):>9}{s(dh):>10}{s(du):>13}{s(dopt):>13}")
    print()

print('=' * 100)
print('TABLE 5 -- THE CONFOUND-CORRECTED HEAD-TO-HEAD')
print('  Compare arms AFTER giving every arm its best scalar rescale (removes the')
print('  "accidental shrinkage" advantage the tracker gets on a bad placement).')
print('=' * 100)
print(f"{'episode':<12}{'A0_on':>9}{'A0_off':>9}{'winner':>10}   "
      f"{'A0_on_opt':>10}{'A0_off_opt':>11}{'winner_opt':>12}")
w_raw, w_opt = [], []
for ep in EPS:
    a, b = rows[('S1-A0', ep)], rows[('X-A0-notrack', ep)]
    wr = 'ON' if a['n'] < b['n'] else 'OFF'
    wo = 'ON' if a['opt'] < b['opt'] else 'OFF'
    w_raw.append(wr); w_opt.append(wo)
    print(f"{SHORT[ep]:<12}{a['n']:9.4f}{b['n']:9.4f}{wr:>10}   "
          f"{a['opt']:10.4f}{b['opt']:11.4f}{wo:>12}")
print(f"raw winner tally  ON {w_raw.count('ON')}/4 | OFF {w_raw.count('OFF')}/4")
print(f"opt winner tally  ON {w_opt.count('ON')}/4 | OFF {w_opt.count('OFF')}/4")

print()
print('=' * 100)
print('TABLE 6 -- per-seed (n=12) sign counts, tracker-ON minus tracker-OFF')
print('   (secondary to the pre-registered median-of-3 unit; reported for texture only)')
print('=' * 100)
for on, off in [('S1-A0', 'X-A0-notrack'), ('S2-T1f', 'X-T1f-notrack')]:
    dn, dr = [], []
    for ep, sd in product(EPS, [42, 123, 456]):
        A = rec[(on, ep)][sd]; B = rec[(off, ep)][sd]
        dn.append(A['nRMSE'] - B['nRMSE'])
        dr.append(A['Pearson_rho'] - B['Pearson_rho'])
    print(f'{on} vs {off}:  d_nRMSE>0 (tracking hurts) {sum(q>0 for q in dn)}/12  '
          f'median {np.median(dn):+.4f}   |   d_rho<0 (tracking hurts) '
          f'{sum(q<0 for q in dr)}/12  median {np.median(dr):+.4f}')

print()
print('=' * 100)
print('TABLE 7 -- init vs tracking: what fraction of the final error is created AFTER init?')
print('=' * 100)
print(f"{'arm':<15}{'init_nrmse':>12}{'track_nRMSE':>13}{'ratio':>8}")
for arm in ['S1-A0', 'X-A0-notrack', 'S2-T1f', 'X-T1f-notrack', 'S2-C1', 'S2-T1']:
    i = np.mean([med(arm, e, 'init_nrmse') for e in EPS])
    t = means[arm]['n'] if arm in means else np.nan
    print(f'{arm:<15}{i:12.4f}{t:13.4f}{t / i:8.2f}')

out = {
    'rows': {f'{a}|{SHORT[e]}': v for (a, e), v in rows.items()},
    'means': means,
}
with open(os.path.join(HERE, 'PRO_shrinkage_arithmetic.json'), 'w') as fh:
    json.dump(out, fh, indent=1, default=float)
print('\nwrote PRO_shrinkage_arithmetic.json')
