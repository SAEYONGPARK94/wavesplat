import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE,'..','s1s2_2026-08-05','results.jsonl'))
        if l.strip() and 'error' not in json.loads(l)]
EPS = sorted({r['episode'] for r in rows})

print('WITHIN-EPISODE correlation of PRIMARY nRMSE with each SECONDARY (n=33 cells/episode):')
print(f'{"episode":<26s}' + ''.join(f'{n:>12s}' for n in ['rho','|VR-1|','Hs_error','amp_rel','skill']))
for ep in EPS:
    rr = [r for r in rows if r['episode'] == ep]
    x = np.array([r['nRMSE'] for r in rr])
    cols = {'rho': [r['Pearson_rho'] for r in rr],
            '|VR-1|': [abs(r['Variance_Ratio']-1) for r in rr],
            'Hs_error': [r['Hs_error'] for r in rr],
            'amp_rel': [r['spec'].get('amp_rel_error_mean', np.nan) for r in rr],
            'skill': [r['Skill_Score'] for r in rr]}
    line = f'{ep[8:-3]:<26s}'
    for k, v in cols.items():
        v = np.array(v, float); m = np.isfinite(v)
        line += f'{np.corrcoef(x[m], v[m])[0,1]:+12.4f}'
    print(line)

print()
print('DOES frame-averaged VR CROSS 1 ?  (the PREREG applies |VR-1| to a frame MEAN,')
print('which is not the mean of |VR-1| unless VR stays on one side of 1)')
lo = [r for r in rows if r['Variance_Ratio'] < 1]; hi = [r for r in rows if r['Variance_Ratio'] >= 1]
print(f'  cells with mean VR < 1: {len(lo)}/132   >= 1: {len(hi)}/132')
import collections
c = collections.Counter(r['arm'] for r in hi)
print(f'  arms with any VR>=1 cell: {dict(c)}')
print('  => for the oracle X-BASE family VR is far below 1 (0.23-0.46), so |mean VR - 1|')
print('     and mean|VR-1| agree there; but any knob that OVERSHOOTS (debate G measured')
print('     VR 1.56-1.72 when both lambdas are zeroed in BOTH phases on the beamform arm)')
print('     breaks the equivalence and the prereg does not say which order applies.')
