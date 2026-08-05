"""CON-3: is the oracle 'placement ceiling' a monotone axis that a method can partially climb?
Uses ONLY today's own s1s2 raw (same ruler, same final-estimate spec block) --
the S1 budget ladder gives 4 non-oracle placement qualities + the oracle end point."""
import json, statistics as st
from itertools import combinations
HERE='/home/vexit/wavesplat/wavesplat/tmp'
EPS=['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
     'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
R={}
for l in open(f'{HERE}/s1s2_2026-08-05/results.jsonl'):
    d=json.loads(l); R.setdefault((d['arm'],d['episode']),{})[d['seed']]=d
def med(arm,ep,path):
    vs=[]
    for v in R[(arm,ep)].values():
        x=v
        for p in path: x=x[p]
        vs.append(x)
    return st.median(vs)
ARMS=['S1-A0','S1-A1','S1-A2','S1-A3','S2-C1','S2-T1','S2-T1f','X-T1f-notrack']
print(f"{'arm':16s} {'nRMSE':>7s} {'det':>6s} {'Edet':>6s} {'k_err':>7s} {'dir':>7s} {'n_est':>6s}")
tab={}
for a in ARMS:
    row=dict(nRMSE=st.mean([med(a,e,['nRMSE']) for e in EPS]),
             det=st.mean([med(a,e,['spec','detection_rate']) for e in EPS]),
             Edet=st.mean([med(a,e,['spec','energy_detection_rate']) for e in EPS]),
             kerr=st.mean([med(a,e,['spec','k_mag_rel_error_mean']) for e in EPS]),
             dirr=st.mean([med(a,e,['spec','direction_error_deg_mean']) for e in EPS]),
             nest=st.mean([med(a,e,['spec','n_est']) for e in EPS]))
    tab[a]=row
    print(f"{a:16s} {row['nRMSE']:7.4f} {row['det']:6.3f} {row['Edet']:6.3f} {row['kerr']:7.3f} {row['dirr']:7.2f} {row['nest']:6.1f}")

print('\n--- Is nRMSE monotone in each placement statistic across the 6 NON-oracle-frozen arms? ---')
non_oracle=['S1-A0','S1-A1','S1-A2','S1-A3','S2-C1']
for k,better in [('det','hi'),('Edet','hi'),('kerr','lo'),('dirr','lo')]:
    conc=disc=0
    for a,b in combinations(non_oracle,2):
        da=tab[a][k]-tab[b][k]; dn=tab[a]['nRMSE']-tab[b]['nRMSE']
        if da==0: continue
        good = (da>0 and dn<0) if better=='hi' else (da>0 and dn>0)
        bad  = (da>0 and dn>0) if better=='hi' else (da>0 and dn<0)
        conc+= good; disc+= bad
    print(f"  {k:6s}: concordant {conc} / discordant {disc}  (Kendall-ish over {len(non_oracle)} arms)")

print('\n--- Decomposition of the 0.2667 -> 0.1671 "oracle placement" gap into its 3 bundled knobs ---')
print(f"  S1-A0 (current)                      {tab['S1-A0']['nRMSE']:.4f}")
print(f"  S2-C1  (N=N_gt only, GT-count)       {tab['S2-C1']['nRMSE']:.4f}   d vs A0 {tab['S2-C1']['nRMSE']-tab['S1-A0']['nRMSE']:+.4f}  <- GT-dependent, HURTS")
print(f"  S2-T1  (true k, k learnable)         {tab['S2-T1']['nRMSE']:.4f}   d vs A0 {tab['S2-T1']['nRMSE']-tab['S1-A0']['nRMSE']:+.4f}  <- true-k value")
print(f"  S2-T1f (true k + k FROZEN)           {tab['S2-T1f']['nRMSE']:.4f}   d vs T1 {tab['S2-T1f']['nRMSE']-tab['S2-T1']['nRMSE']:+.4f}  <- k-freeze knob (GT-FREE)")
print(f"  X-T1f-notrack (+ tracker off)        {tab['X-T1f-notrack']['nRMSE']:.4f}   d vs T1f {tab['X-T1f-notrack']['nRMSE']-tab['S2-T1f']['nRMSE']:+.4f}  <- tracker-off knob (GT-FREE)")
tot=tab['S1-A0']['nRMSE']-tab['X-T1f-notrack']['nRMSE']
gtdep=tab['S1-A0']['nRMSE']-tab['S2-T1']['nRMSE']
print(f"\n  total A0 -> oracle+notrack        {tot:.4f}")
print(f"  attributable to TRUE-K (GT-only)  {gtdep:.4f}  = {100*gtdep/tot:.1f}%")
print(f"  attributable to GT-FREE knobs     {tot-gtdep:.4f}  = {100*(tot-gtdep)/tot:.1f}%"
      "   (k-freeze + tracker-off, both deployable)")
print(f"\n  but the same GT-free knobs on the CURRENT placement: X-A0-notrack "
      f"{st.mean([med('X-A0-notrack',e,['nRMSE']) for e in EPS]):.4f} vs A0 {tab['S1-A0']['nRMSE']:.4f}"
      "  -> tracker-off ALONE is WORSE. Knobs pay only when placement is right.")
json.dump(tab,open(f'{HERE}/waveprior_debate_2026-08-05/CON_monotonicity.json','w'),indent=1)
