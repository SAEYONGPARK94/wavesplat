"""CON probe — attack line 3 (stratum artifact) and attack line 4 (pairing audit),
plus the 2x2 identity check."""
import json
import numpy as np
from collections import defaultdict

rows = [json.loads(l) for l in open('tmp/s1s2_2026-08-05/results.jsonl')]
c = defaultdict(list)
for r in rows: c[(r['arm'], r['episode'])].append(r)
eps = ['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
def m(a,e,k): return float(np.median([r[k] for r in c[(a,e)]]))
def spread(a,e,k):
    v=[r[k] for r in c[(a,e)]]; return max(v)-min(v)

print('='*112)
print('C1 — PER-EPISODE tracker effect at BOTH placements, against the per-cell seed spread')
print('='*112)
print(f'{"episode":22s}{"n_gt":>5s} | {"DEPLOYED (S1-A0)":>34s} | {"ORACLE (S2-T1f)":>34s}')
print(f'{"":22s}{"":>5s} | {"init":>7s}{"ON":>8s}{"OFF":>8s}{"harm":>8s}{"noise":>8s} | '
      f'{"init":>7s}{"ON":>8s}{"OFF":>8s}{"harm":>8s}{"noise":>8s}')
for e in eps:
    n_gt = 15 if '15comp' in e else 30
    dep = (m('S1-A0',e,'init_nrmse'), m('S1-A0',e,'nRMSE'), m('X-A0-notrack',e,'nRMSE'))
    orc = (m('S2-T1f',e,'init_nrmse'), m('S2-T1f',e,'nRMSE'), m('X-T1f-notrack',e,'nRMSE'))
    nd = max(spread('S1-A0',e,'nRMSE'), spread('X-A0-notrack',e,'nRMSE'))
    no = max(spread('S2-T1f',e,'nRMSE'), spread('X-T1f-notrack',e,'nRMSE'))
    print(f'{e[8:30]:22s}{n_gt:5d} | {dep[0]:7.4f}{dep[1]:8.4f}{dep[2]:8.4f}{dep[1]-dep[2]:+8.4f}{nd:8.4f} | '
          f'{orc[0]:7.4f}{orc[1]:8.4f}{orc[2]:8.4f}{orc[1]-orc[2]:+8.4f}{no:8.4f}')
print('  harm = tracked_ON - tracked_OFF  (positive = tracker hurts); noise = within-cell seed range')

print()
print('  |harm| / seed-noise ratio:')
for e in eps:
    hd = abs(m('S1-A0',e,'nRMSE')-m('X-A0-notrack',e,'nRMSE'))
    ho = abs(m('S2-T1f',e,'nRMSE')-m('X-T1f-notrack',e,'nRMSE'))
    nd = max(spread('S1-A0',e,'nRMSE'), spread('X-A0-notrack',e,'nRMSE'))
    no = max(spread('S2-T1f',e,'nRMSE'), spread('X-T1f-notrack',e,'nRMSE'))
    print(f'   {e[8:30]:24s} deployed {hd/nd:5.2f}x   oracle {ho/no:5.2f}x')

print()
print('='*112)
print('C2 — the "tracker eats 52% of the placement gain" 2x2 IS the contraction factor')
print('='*112)
onA  = np.mean([m('S1-A0',e,'nRMSE') for e in eps]);  onT = np.mean([m('S2-T1f',e,'nRMSE') for e in eps])
offA = np.mean([m('X-A0-notrack',e,'nRMSE') for e in eps]); offT = np.mean([m('X-T1f-notrack',e,'nRMSE') for e in eps])
iA   = np.mean([m('S1-A0',e,'init_nrmse') for e in eps]);   iT  = np.mean([m('S2-T1f',e,'init_nrmse') for e in eps])
print(f'  placement gain, tracker ON  = {onA-onT:.4f}')
print(f'  placement gain, tracker OFF = {offA-offT:.4f}   (init gain = {iA-iT:.4f}, i.e. the OFF row IS init)')
print(f'  surviving fraction = {(onA-onT)/(offA-offT):.4f}   vs fitted contraction b = 0.5639')
print('  => "the tracker eats 52%" and "b = 0.56" are the same number, not two findings.')

print()
print('='*112)
print('C3 — PAIRING AUDIT (attack line 4)')
print('='*112)
ok = True
for a_on, a_off in [('S1-A0','X-A0-notrack'), ('S2-T1f','X-T1f-notrack')]:
    for e in eps:
        s_on = sorted(r['seed'] for r in c[(a_on,e)]); s_off = sorted(r['seed'] for r in c[(a_off,e)])
        n_on = sorted(r['N'] for r in c[(a_on,e)]);    n_off = sorted(r['N'] for r in c[(a_off,e)])
        i_on = sorted(round(r['init_nrmse'],9) for r in c[(a_on,e)])
        i_off= sorted(round(r['init_nrmse'],9) for r in c[(a_off,e)])
        good = (s_on==s_off) and (n_on==n_off) and (i_on==i_off)
        ok &= good
        print(f'  {a_on:8s}/{a_off:14s} {e[8:26]:20s} seeds_match={s_on==s_off} '
              f'N_match={n_on==n_off} init_bit_identical={i_on==i_off}')
print(f'  ==> ON/OFF pairs share seed, N and a BIT-IDENTICAL initialisation: {ok}')
print('      (the tracker contrast itself is clean; the confound is BETWEEN placements, next)')
print()
print('  BETWEEN-placement confound: the "placement" factor of the 2x2 also changes model order')
for e in eps:
    print(f'   {e[8:26]:20s} N(A0)={m("S1-A0",e,"N"):5.1f}  N(T1f)={m("S2-T1f",e,"N"):5.1f}  '
          f'ratio={m("S2-T1f",e,"N")/m("S1-A0",e,"N"):.2f}x   '
          f'oracle harm={m("S2-T1f",e,"nRMSE")-m("X-T1f-notrack",e,"nRMSE"):+.4f}')
r_ = np.corrcoef([m('S2-T1f',e,'N') for e in eps],
                 [m('S2-T1f',e,'nRMSE')-m('X-T1f-notrack',e,'nRMSE') for e in eps])[0,1]
print(f'   r(N, oracle harm) over the 4 episodes = {r_:+.3f}  '
      f'(n=4; harm is NOT increasing in N -> model order does not explain the harm)')

print()
print('='*112)
print('C4 — what a PERFECT amplitude fix can buy (nRMSE = s*sqrt(1-2r*sqrt(v)+v), s=std_gt/Hs)')
print('='*112)
print(f'{"arm":16s}{"ep":22s}{"rho":>8s}{"VR":>8s}{"nRMSE":>9s}{"opt-rescale":>13s}{"recovery":>10s}')
for arm in ['S2-T1f','S1-A0']:
    for e in eps:
        rho, vr, nr = m(arm,e,'Pearson_rho'), m(arm,e,'Variance_Ratio'), m(arm,e,'nRMSE')
        s = nr/np.sqrt(max(1e-9, 1-2*rho*np.sqrt(vr)+vr))
        opt = s*np.sqrt(max(0.0,1-rho**2))
        print(f'{arm:16s}{e[8:28]:22s}{rho:8.4f}{vr:8.4f}{nr:9.4f}{opt:13.4f}{(nr-opt)/nr*100:9.1f}%')
print('  recovery = fraction of the arm\'s own nRMSE that PERFECT amplitude rescaling removes.')
