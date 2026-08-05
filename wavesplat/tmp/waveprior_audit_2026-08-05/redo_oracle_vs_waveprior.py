"""Item (6): re-read WavePrior's crowded gain against today's oracle-(kx,ky) ceiling,
on the FOUR episodes both experiments share. Existing raw only, no new runs.
"""
import json
import numpy as np
from collections import defaultdict

S = '/home/vexit/wavesplat/wavesplat/tmp/s1s2_2026-08-05/results.jsonl'
I = '/home/vexit/wavesplat/wavesplat/tmp/wp_integration_2026-07-25/arm_runs.jsonl'
C = '/home/vexit/wavesplat/wavesplat/tmp/wp_crowded_2026-07-25/cond_runs.jsonl'

EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

s = defaultdict(lambda: defaultdict(list))
for l in open(S):
    r = json.loads(l); s[r['episode']][r['arm']].append(r['nRMSE'])
i = defaultdict(lambda: defaultdict(list))
for l in open(I):
    r = json.loads(l); i[r['episode']][r['arm']].append(r['avg_nRMSE'])
c = defaultdict(list)
for l in open(C):
    r = json.loads(l)
    if r['cond_polish_nrmse_Hs'] is not None: c[r['episode']].append(r['cond_polish_nrmse_Hs'])

print('NOTE: s1s2 = median-of-3 seeds {42,123,456}; wp = median-of-5 lanes. Same 4 episodes,')
print('      same data_finemesh_2026-07-23, same nRMSE/Hs metric family. NOT a re-run: read-across.\n')
hdr = ('episode', 'WS A0', 'v2_konly', 'WP learn', 'WP cond', 'S2-T1f', 'LETKF')
print('%-26s %8s %9s %9s %8s %8s' % hdr[:6])
LETKF = {'episode_BF4_15comp_seed0_uniform.h5': 0.2282, 'episode_BF4_30comp_seed0_uniform.h5': 0.2347,
         'episode_BF5_15comp_seed0_uniform.h5': 0.2298, 'episode_BF5_30comp_seed0_uniform.h5': 0.2651}
tab = {}
for e in EPS:
    row = dict(a0=np.median(s[e]['S1-A0']), t1f=np.median(s[e]['S2-T1f']), t1=np.median(s[e]['S2-T1']),
               c1=np.median(s[e]['S2-C1']),
               v2=np.median(i[e]['v2_konly']), lrn=np.median(i[e]['learned']),
               lrnk=np.median(i[e]['learned_konly']), cond=np.median(c[e]))
    tab[e] = row
    print('%-26s %8.4f %9.4f %9.4f %8.4f %8.4f %8.4f' %
          (e[8:-3], row['a0'], row['v2'], row['lrn'], row['cond'], row['t1f'], LETKF[e]))
m = {k: np.mean([tab[e][k] for e in EPS]) for k in tab[EPS[0]]}
print('%-26s %8.4f %9.4f %9.4f %8.4f %8.4f %8.4f' %
      ('MEAN4', m['a0'], m['v2'], m['lrn'], m['cond'], m['t1f'], np.mean(list(LETKF.values()))))

print('\n--- placement-gain accounting on these 4 episodes (baseline = v2_konly tracked) ---')
base = m['v2']
for lab, k in (('WP learned (tracked)', 'lrn'), ('WP learned_konly (k only)', 'lrnk'),
               ('WP cond (proposal-level nowcast)', 'cond'), ('oracle k, frozen (S2-T1f)', 't1f'),
               ('oracle k, learnable (S2-T1)', 't1')):
    g = base - m[k]
    print('  %-34s %.4f  gain %.4f  = %5.1f%% of the oracle-k gain' %
          (lab, m[k], g, 100 * g / (base - m['t1f'])))
print('  %-34s %.4f  gain %.4f' % ('S1-A0 (current shipped init)', m['a0'], base - m['a0']))
print('\n  oracle-k gain (v2_konly -> S2-T1f) = %.4f ; WP learned captures %.1f%% of it'
      % (base - m['t1f'], 100 * (base - m['lrn']) / (base - m['t1f'])))
print('  residual left on the table after WP = %.4f nRMSE/Hs (%.0f%% of the WP gain again)'
      % (m['lrn'] - m['t1f'], 100 * (m['lrn'] - m['t1f']) / (base - m['lrn'])))
