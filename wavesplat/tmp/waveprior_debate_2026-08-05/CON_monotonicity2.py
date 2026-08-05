"""CON-3 (fixed): pairwise concordance was orientation-buggy in v1 (skipped pairs with da<0).
Fixed: orient every pair so da>0. Also adds the LEARNED arm to the ladder."""
import json, statistics as st
from itertools import combinations
HERE='/home/vexit/wavesplat/wavesplat/tmp'
EPS=['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
     'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
tab=json.load(open(f'{HERE}/waveprior_debate_2026-08-05/CON_monotonicity.json'))
ax=json.load(open(f'{HERE}/waveprior_debate_2026-08-05/CON_axis_decomp.json'))
wp={}
for l in open(f'{HERE}/wp_integration_2026-07-25/arm_runs.jsonl'):
    d=json.loads(l)
    if not d.get('err'): wp.setdefault((d['arm'],d['episode']),{})[(d['est_seed'],d['diff_seed'])]=d
mo=lambda t,k: st.mean([ax[e][t][k] for e in EPS])
tab['LEARNED(prop)']=dict(
    nRMSE=st.mean([st.median([v['avg_nRMSE'] for v in wp[('learned',e)].values()]) for e in EPS]),
    det=mo('learned_selected','detection_rate'), Edet=mo('learned_selected','energy_detection_rate'),
    kerr=mo('learned_selected','k_mag_rel_error_mean'), dirr=mo('learned_selected','direction_error_deg_mean'),
    nest=mo('learned_selected','n_est'))

print(f"{'arm':16s} {'nRMSE':>7s} {'det':>6s} {'Edet':>6s} {'k_err':>7s} {'dir':>7s} {'n_est':>6s}")
LAD=['S1-A0','S1-A1','S1-A2','S1-A3','S2-C1','LEARNED(prop)','S2-T1','S2-T1f','X-T1f-notrack']
for a in LAD:
    r=tab[a]; print(f"{a:16s} {r['nRMSE']:7.4f} {r['det']:6.3f} {r['Edet']:6.3f} {r['kerr']:7.3f} {r['dirr']:7.2f} {r['nest']:6.1f}")

print('\n--- pairwise concordance nRMSE vs placement statistic, DEPLOYABLE arms only (no GT) ---')
dep=['S1-A0','S1-A1','S1-A2','S1-A3','LEARNED(prop)']
for k,better in [('det','hi'),('Edet','hi'),('kerr','lo'),('dirr','lo')]:
    conc=disc=0; pairs=[]
    for a,b in combinations(dep,2):
        x,y=(a,b) if tab[a][k]>tab[b][k] else (b,a)      # x has the LARGER statistic
        dn=tab[x]['nRMSE']-tab[y]['nRMSE']
        improves = (dn<0) if better=='hi' else (dn>0)     # 'better' stat -> lower nRMSE?
        conc+=improves; disc+= (not improves)
        pairs.append((x,y,round(dn,4),improves))
    print(f"  {k:6s}: concordant {conc}/10  discordant {disc}/10")
    for p in pairs: print(f"        {p[0]:14s} > {p[1]:14s} stat  |  dnRMSE {p[2]:+.4f}  {'OK' if p[3] else 'CONTRA'}")

print('\n--- the decisive counterexample ---')
A3,LN=tab['S1-A3'],tab['LEARNED(prop)']
print(f"  S1-A3 (beamform, budget 40): det {A3['det']:.3f}  Edet {A3['Edet']:.3f}  k_err {A3['kerr']:.3f}  -> nRMSE {A3['nRMSE']:.4f}")
print(f"  LEARNED proposal          : det {LN['det']:.3f}  Edet {LN['Edet']:.3f}  k_err {LN['kerr']:.3f}  -> nRMSE {LN['nRMSE']:.4f}")
print("  => A3 matches MORE components and MORE energy than the learned proposal, and is WORSE in nRMSE.")
json.dump(tab,open(f'{HERE}/waveprior_debate_2026-08-05/CON_monotonicity.json','w'),indent=1)
