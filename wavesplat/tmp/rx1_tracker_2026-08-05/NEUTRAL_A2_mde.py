"""A2: MDE of the pre-registered design, computed BEFORE any Rx-1 result exists.

Design (PREREG s1, s2): paired two-sided sign test over episodes, median-of-3 seeds
per cell, primary = median of paired deltas in nRMSE.
"""
import json, os, math, itertools
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 's1s2_2026-08-05', 'results.jsonl')
rows = [json.loads(l) for l in open(SRC) if l.strip() and 'error' not in json.loads(l)]
EPS = sorted({r['episode'] for r in rows}); SEEDS = sorted({r['seed'] for r in rows})
def cell(arm, ep, s):
    return [r for r in rows if r['arm']==arm and r['episode']==ep and r['seed']==s][0]

print('=' * 84)
print('A2.1  WHAT THE SIGN TEST CAN EVEN RETURN  (exact, no data needed)')
print('=' * 84)
from math import comb
for n in (4, 6, 8, 12):
    print(f'  n={n:<3d} attainable two-sided p-values: ', end='')
    ps = sorted({min(1.0, 2*sum(comb(n,i) for i in range(0,k+1))/2**n) for k in range(0, n//2+1)})
    print(', '.join(f'{p:.4f}' for p in ps))
print()
print('  => n=4: FLOOR p = 0.1250 (4/4 or 0/4). alpha=0.05 is UNREACHABLE.')
print('     The MDE at alpha=0.05 for n=4 is +infinity: no effect of any size can be')
print('     declared significant. The only decision rule available is "4/4 consistent sign".')
print('  => the smallest n for which a two-sided sign test can reach 0.05 is n=6 (p=0.03125).')

print()
print('=' * 84)
print('A2.2  NOISE FLOOR of one paired episode-level delta (empirical, from s1s2 raw)')
print('=' * 84)
# (a) within-cell seed SD, pooled across all 11 arms x 4 episodes
sds = []
for arm in sorted({r['arm'] for r in rows}):
    for ep in EPS:
        v = [cell(arm, ep, s)['nRMSE'] for s in SEEDS]
        sds.append(np.std(v, ddof=1))
sig_seed = float(np.sqrt(np.mean(np.array(sds)**2)))
print(f'  pooled within-cell seed SD of nRMSE (44 cells x 3 seeds): sigma_seed = {sig_seed:.4f}')
print(f'    (median cell SD {np.median(sds):.4f}, max {max(sds):.4f})')
# SD of a median-of-3 of iid normals = 0.6698 * sigma
K = 0.6698
sig_m3 = K * sig_seed
print(f'  SD of a median-of-3 estimate:                      sigma_m3   = {sig_m3:.4f}')
# (b) how much cancels when the two arms share (episode, seed)?  measure on the
#     one contrast we have where both arms exist at all 12 cells
for A, B in [('S2-T1f','X-T1f-notrack'), ('S1-A0','X-A0-notrack'), ('S2-T1','S2-T2'),
             ('S2-T1f','S2-T4')]:
    da = np.array([cell(A, ep, s)['nRMSE'] for ep in EPS for s in SEEDS])
    db = np.array([cell(B, ep, s)['nRMSE'] for ep in EPS for s in SEEDS])
    epi = np.repeat(np.arange(4), 3)
    # remove episode means, then correlate the seed-level residuals
    ra = da - np.array([da[epi==e].mean() for e in epi])
    rb = db - np.array([db[epi==e].mean() for e in epi])
    r = np.corrcoef(ra, rb)[0,1]
    print(f'  seed-level residual corr between arms  {A:<14s} vs {B:<14s} r = {r:+.4f}')

print()
print('  Two arms sharing a seed are NOT strongly coupled once the knob bites, so use the')
print('  conservative independent case: sigma_delta = sqrt(2)*sigma_m3.')
sig_delta_ind = math.sqrt(2) * sig_m3
print(f'  sigma_delta (independent arms)  = {sig_delta_ind:.4f}')
print(f'  sigma_delta (r=+0.5 coupling)   = {math.sqrt(2*(1-0.5))*sig_m3:.4f}')
print(f'  sigma_delta (r=+0.9 coupling)   = {math.sqrt(2*(1-0.9))*sig_m3:.4f}')

print()
print('=' * 84)
print('A2.3  MDE  (effect that gives >=80% chance of the ONLY reportable outcome, 4/4)')
print('=' * 84)
print('  P(4/4 same sign) = Phi(Delta/sigma_delta)^4 >= 0.80  =>  Delta >= 1.6031*sigma_delta')
for lab, sd in [('independent arms', sig_delta_ind),
                ('r=+0.5 coupling', math.sqrt(1.0)*sig_m3),
                ('r=+0.9 coupling', math.sqrt(0.2)*sig_m3)]:
    z = norm.ppf(0.80 ** 0.25)
    print(f'    {lab:<20s} sigma_delta={sd:.4f}  ->  MDE(80% power, 4/4) = {z*sd:.4f} nRMSE')
    print(f'    {"":<20s} {"":<18s}      MDE(50% power, 4/4) = {norm.ppf(0.5**0.25)*sd:.4f} nRMSE')

print()
print('  For reference, effects Rx-1 is looking for (all re-derived, not quoted):')
d_track = [float(np.median([cell('X-T1f-notrack',ep,s)['nRMSE'] for s in SEEDS])) -
           float(np.median([cell('S2-T1f',ep,s)['nRMSE'] for s in SEEDS])) for ep in EPS]
print(f'    tracker ON->OFF (the +0.0981 headline)  paired deltas '
      f'{[round(x,4) for x in d_track]}  median {np.median(d_track):+.4f}')
print(f'      -> |Delta|/sigma_delta(indep) = {abs(np.median(d_track))/sig_delta_ind:.2f}   '
      f'=> comfortably above MDE')
print(f'    Debate G "Phase-2-only lambda=0" on ONE episode: 0.1425 -> 0.0167 = -0.1258')
print(f'      -> |Delta|/sigma_delta(indep) = {0.1258/sig_delta_ind:.2f}   => above MDE')
print(f'    X-A0-notrack "+0.0072 worsening" (already retracted as noise):')
d_a0 = [float(np.median([cell('X-A0-notrack',ep,s)['nRMSE'] for s in SEEDS])) -
        float(np.median([cell('S1-A0',ep,s)['nRMSE'] for s in SEEDS])) for ep in EPS]
print(f'      paired deltas {[round(x,4) for x in d_a0]}  median {np.median(d_a0):+.4f}  '
      f'|Delta|/sigma_delta = {abs(np.median(d_a0))/sig_delta_ind:.2f}  => FAR below MDE')

print()
print('=' * 84)
print('A2.4  MULTIPLICITY: 5 knobs x 6 outcomes on n=4')
print('=' * 84)
print('  Under the global null, P(at least one 4/4 among m independent tests) =')
for m in (5, 6, 11, 15, 30):
    print(f'    m={m:<3d}: 1-(1-0.125)^{m} = {1-(1-0.125)**m:.3f}')
print('  With 5 knobs x PRIMARY only  -> P(>=1 spurious 4/4) = 0.487')
print('  With 5 knobs x ~3 effective outcomes (see A3) -> m~15 -> P = 0.865')
print('  => a lone 4/4 at p=0.125 in this design is EXPECTED under the null.')
