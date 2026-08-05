"""CON-7: the 'recovered a fraction of the ceiling' number is metric-dependent.
Matched 4 episodes only. v2 baseline & learned from wp_integration (07-25),
oracle from s1s2 (08-05). The 07-25/08-05 v2 lanes are bit-identical at seed 42 (CON_era)."""
import json, statistics as st
HERE='/home/vexit/wavesplat/wavesplat/tmp'
EPS=['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
     'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
wp={}
for l in open(f'{HERE}/wp_integration_2026-07-25/arm_runs.jsonl'):
    d=json.loads(l)
    if not d.get('err'): wp.setdefault(d['arm'],{}).setdefault(d['episode'],{})[(d['est_seed'],d['diff_seed'])]=d
s1={}
for l in open(f'{HERE}/s1s2_2026-08-05/results.jsonl'):
    d=json.loads(l); s1.setdefault(d['arm'],{}).setdefault(d['episode'],{})[d['seed']]=d
MAP=[('nRMSE','avg_nRMSE','lo'),('Pearson_rho','avg_Pearson_rho','hi'),
     ('Skill_Score','avg_Skill_Score','hi'),('Directional_R2','avg_Directional_R2','hi'),
     ('Variance_Ratio','avg_Variance_Ratio','one'),('Hs_error','avg_Hs_error','lo')]
print(f"{'metric':16s} {'v2':>8s} {'learned':>8s} {'orclTRK':>8s} {'orcl+notrk':>10s} "
      f"{'rec% vs TRK':>11s} {'rec% vs notrk':>13s}")
out={}
for sk,wk,dirn in MAP:
    v2=st.mean([st.median([v[wk] for v in wp['v2_konly'][e].values()]) for e in EPS])
    ln=st.mean([st.median([v[wk] for v in wp['learned'][e].values()]) for e in EPS])
    o1=st.mean([st.median([v[sk] for v in s1['S2-T1f'][e].values()]) for e in EPS])
    o2=st.mean([st.median([v[sk] for v in s1['X-T1f-notrack'][e].values()]) for e in EPS])
    if dirn=='one':   # VR: distance from 1
        f=lambda x: abs(x-1.0); v2f,lnf,o1f,o2f=f(v2),f(ln),f(o1),f(o2)
    else:
        v2f,lnf,o1f,o2f=v2,ln,o1,o2
    r1=100*(v2f-lnf)/(v2f-o1f) if v2f!=o1f else float('nan')
    r2=100*(v2f-lnf)/(v2f-o2f) if v2f!=o2f else float('nan')
    out[sk]=dict(v2=v2,learned=ln,oracle_trk=o1,oracle_notrk=o2,rec_trk=r1,rec_notrk=r2)
    print(f"{sk:16s} {v2:8.4f} {ln:8.4f} {o1:8.4f} {o2:10.4f} {r1:10.1f}% {r2:12.1f}%")
print("\n  (VR scored as |VR-1|. 'rec%' = (v2 - learned)/(v2 - oracle) on the SAME 4 episodes.)")
print("  spread of the recovery fraction across metrics:")
for tag,k in [('vs oracle+tracker','rec_trk'),('vs oracle+notrack','rec_notrk')]:
    vals=[out[m][k] for m,_,_ in MAP]
    print(f"    {tag:20s} min {min(vals):.1f}%  max {max(vals):.1f}%  spread {max(vals)-min(vals):.1f} points")
json.dump(out,open(f'{HERE}/waveprior_debate_2026-08-05/CON_recovery_by_metric.json','w'),indent=1)
